"""
Monkey-patch mechanism that makes transformers' `DynamicCache` layers store
(and return) uniformly N-bit fake-quantized key/value states, for the
"baseline-uniform-2bit" experiment.

DESIGN NOTE — why a class-level monkeypatch of `update()`, and not a registry
swap or per-instance override:

transformers.cache_utils.DynamicCache builds its per-layer cache objects from
a module-level global dict `LAYER_TYPE_CACHE_MAPPING: dict[str, type]`
(e.g. {"full_attention": DynamicLayer, "sliding_attention":
DynamicSlidingWindowLayer, ...}). Gemma-2 alternates "sliding_attention" and
"full_attention" per layer, so a `DynamicCache` built for it ends up with a
mix of `DynamicLayer` and `DynamicSlidingWindowLayer` instances, dispatched
through that registry at `DynamicCache.__init__` time.

Two ways to intercept this were considered:
  1. Mutate `LAYER_TYPE_CACHE_MAPPING` itself (e.g. point "full_attention" at
     a quantizing subclass). Rejected: this dict is process-global and shared
     by every model that constructs a `DynamicCache` in this process — on a
     shared research-platform process, mutating it would silently change the
     cache behaviour of unrelated models/experiments running in the same
     process, which is a correctness hazard.
  2. Monkeypatch `update()` at the CLASS level directly on `DynamicLayer` and
     `DynamicSlidingWindowLayer` (this module's approach). This is hit
     regardless of how the layer instance was constructed — whether via the
     registry dispatch above, or any code path that falls back to
     `DynamicLayer` directly — because both classes are patched in place.
     It also does not touch the registry dict, so cache *construction*
     behaviour (which class is chosen for which layer_type) is completely
     unaffected; only what `update()` does with the tensors changes.

TRADEOFF: because the patch operates at the class level, once installed it
affects ALL models/instances in the process that use `DynamicLayer` /
`DynamicSlidingWindowLayer` for the duration the patch is installed — not
just the `model` object passed to `install_uniform_kv_quant`. This is
documented explicitly, and `uninstall_uniform_kv_quant()` (plus the
`uniform_kv_quant` context manager below) is provided so the effect is
strictly scoped and reversible: install immediately before the
generate/forward calls you want quantized, uninstall (or exit the context
manager) immediately after.
"""

import functools

from transformers.cache_utils import DynamicLayer, DynamicSlidingWindowLayer

from quantization import quantize_dequantize

# Keep references to the ORIGINAL unbound `update` functions at module load
# time, before any patching happens, so we can always restore them exactly.
_ORIG_DYNAMIC_UPDATE = DynamicLayer.update
_ORIG_SLIDING_UPDATE = DynamicSlidingWindowLayer.update

# Module-level sentinel guarding against double-patching.
_PATCH_STATE = {"installed": False, "num_bits": None}


def _make_quantizing_update(orig_update_fn, num_bits: int):
    """
    Build a replacement `update(self, key_states, value_states, *args,
    **kwargs)` method that fake-quantizes `key_states`/`value_states` BEFORE
    delegating to `orig_update_fn` (the original class's `update`).

    Because the quantized tensors are substituted for the raw ones prior to
    calling into the original concatenation logic, both:
      (i)  what gets written into the persistent cache (`self.keys`/
           `self.values`), and
      (ii) what is returned for this step's attention computation,
    are quantized identically — there is no discrepancy between "what's
    cached" and "what's used to attend" at this step.
    """

    @functools.wraps(orig_update_fn)
    def _quantizing_update(self, key_states, value_states, *args, **kwargs):
        key_states_q = quantize_dequantize(
            key_states, num_bits=num_bits, group_size=None, axis=-1
        )
        value_states_q = quantize_dequantize(
            value_states, num_bits=num_bits, group_size=None, axis=-1
        )
        return orig_update_fn(self, key_states_q, value_states_q, *args, **kwargs)

    return _quantizing_update


def install_uniform_kv_quant(model, num_bits: int = 2) -> None:
    """
    Install the uniform N-bit fake-quantization patch on the KV cache update
    path used by `DynamicCache` (`DynamicLayer.update` and
    `DynamicSlidingWindowLayer.update`), globally, at the class level.

    This is the only reliable interception point available without mutating
    the shared `LAYER_TYPE_CACHE_MAPPING` registry (see module docstring for
    the full rationale). Because the patch is installed on the *classes*,
    not on `model` itself, it affects every model in the process using these
    cache layer classes for as long as it remains installed — `model` is
    accepted here for API symmetry (so call sites read naturally as "quantize
    this model's cache") and so its class/config can be logged for
    auditability, but the current implementation does not — and structurally
    cannot, given transformers' cache construction — scope the patch to a
    single model instance. Call `uninstall_uniform_kv_quant()` (or use the
    `uniform_kv_quant` context manager below) to scope its effect in time.

    Args:
        model: the model whose KV cache we intend to quantize. Not mutated
            directly; used only for logging/identification.
        num_bits: number of bits for fake-quantization (default 2, matching
            this experiment's uniform-2-bit baseline).
    """
    if _PATCH_STATE["installed"]:
        raise RuntimeError(
            "install_uniform_kv_quant() called while a patch is already "
            f"installed (num_bits={_PATCH_STATE['num_bits']}). Call "
            "uninstall_uniform_kv_quant() first, or use the "
            "uniform_kv_quant() context manager to avoid double-patching."
        )

    model_id = getattr(model, "name_or_path", None) or type(model).__name__
    print(
        f"[cache_patch] installing uniform {num_bits}-bit KV fake-quant patch "
        f"(class-level, global) for model={model_id!r}"
    )

    DynamicLayer.update = _make_quantizing_update(_ORIG_DYNAMIC_UPDATE, num_bits)
    DynamicSlidingWindowLayer.update = _make_quantizing_update(
        _ORIG_SLIDING_UPDATE, num_bits
    )

    _PATCH_STATE["installed"] = True
    _PATCH_STATE["num_bits"] = num_bits


def uninstall_uniform_kv_quant() -> None:
    """
    Restore the original (unquantized) `update` methods on `DynamicLayer` and
    `DynamicSlidingWindowLayer`. Safe to call even if no patch is currently
    installed (no-op in that case).
    """
    if not _PATCH_STATE["installed"]:
        return

    DynamicLayer.update = _ORIG_DYNAMIC_UPDATE
    DynamicSlidingWindowLayer.update = _ORIG_SLIDING_UPDATE

    print(
        f"[cache_patch] uninstalled uniform {_PATCH_STATE['num_bits']}-bit KV "
        "fake-quant patch, restored original DynamicLayer/DynamicSlidingWindowLayer.update"
    )

    _PATCH_STATE["installed"] = False
    _PATCH_STATE["num_bits"] = None


class uniform_kv_quant:
    """
    Context manager wrapping install/uninstall in try/finally, so the patch's
    effect is always scoped and reversible even if an exception is raised
    inside the `with` block.

    Usage:
        with uniform_kv_quant(model, num_bits=2):
            outputs = model.generate(**inputs)
    """

    def __init__(self, model, num_bits: int = 2):
        self.model = model
        self.num_bits = num_bits

    def __enter__(self):
        install_uniform_kv_quant(self.model, num_bits=self.num_bits)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        uninstall_uniform_kv_quant()
        return False
