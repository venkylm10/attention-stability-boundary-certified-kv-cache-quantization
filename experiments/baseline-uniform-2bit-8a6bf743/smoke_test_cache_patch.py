"""
SMOKE TEST ONLY — validates the patch mechanism (install_uniform_kv_quant /
uninstall_uniform_kv_quant from cache_patch.py) on an ungated proxy model;
does NOT touch the pinned google/gemma-2-9b-it and its output is NOT an
experiment result. This file exists purely to demonstrate that the class-
level DynamicLayer/DynamicSlidingWindowLayer monkeypatch (a) actually alters
generation numerically when installed, and (b) is cleanly reversible.

Candidate proxy models are tried in order; the first one that both (i)
downloads/loads successfully and (ii) is confirmed to use `DynamicCache`
with `DynamicLayer`/`DynamicSlidingWindowLayer` instances is used. Plain
`gpt2` is deliberately excluded as a fallback because it uses the legacy
tuple-based cache, not `DynamicCache` — patching would be a no-op for it.
"""

import sys
import traceback

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache, DynamicLayer, DynamicSlidingWindowLayer

from cache_patch import install_uniform_kv_quant, uninstall_uniform_kv_quant

CANDIDATE_MODELS = [
    "hf-internal-testing/tiny-random-Gemma2ForCausalLM",
    "hf-internal-testing/tiny-random-LlamaForCausalLM",
    "hf-internal-testing/tiny-random-Gemma3ForCausalLM",
]

PROMPT = "The quick brown fox"
MAX_NEW_TOKENS = 8


def _load_candidate(model_name: str):
    print(f"[smoke_test] attempting to load candidate proxy model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    return tokenizer, model, device


def _verify_uses_dynamic_cache(model, tokenizer, device) -> bool:
    """Run a tiny forward pass and confirm past_key_values is a DynamicCache
    built from DynamicLayer/DynamicSlidingWindowLayer instances."""
    inputs = tokenizer(PROMPT, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs, use_cache=True)
    pkv = out.past_key_values
    print(f"[smoke_test]   type(past_key_values) = {type(pkv)}")
    if not isinstance(pkv, DynamicCache):
        return False
    layer_types_ok = all(
        isinstance(layer, (DynamicLayer, DynamicSlidingWindowLayer))
        for layer in pkv.layers
    )
    print(f"[smoke_test]   all layers are DynamicLayer/DynamicSlidingWindowLayer: {layer_types_ok}")
    return layer_types_ok


def run_smoke_test():
    tokenizer = None
    model = None
    device = None
    used_model_name = None
    load_errors = []

    for name in CANDIDATE_MODELS:
        try:
            tokenizer, model, device = _load_candidate(name)
            if _verify_uses_dynamic_cache(model, tokenizer, device):
                used_model_name = name
                print(f"[smoke_test] SELECTED proxy model: {name} (confirmed DynamicCache/DynamicLayer usage)")
                break
            else:
                print(f"[smoke_test] REJECTED {name}: does not use DynamicCache with DynamicLayer instances")
                model = None
        except Exception as e:  # noqa: BLE001 - want to try next candidate on any failure
            load_errors.append((name, repr(e)))
            print(f"[smoke_test] FAILED to load {name}: {e}")
            traceback.print_exc(limit=2)

    if model is None:
        print("[smoke_test] No valid proxy model could be loaded/verified. Tried:")
        for name, err in load_errors:
            print(f"    - {name}: {err}")
        print("[FAIL] smoke test could not run: no proxy model available")
        return False

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(PROMPT, return_tensors="pt").to(device)
    gen_kwargs = dict(
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        num_beams=1,
    )

    # 1. Baseline generation, unpatched.
    with torch.no_grad():
        baseline_out = model.generate(**inputs, **gen_kwargs)
    baseline_ids = baseline_out[0].tolist()
    print(f"[smoke_test] baseline (unpatched) generated ids: {baseline_ids}")

    # 2. Install the patch, generate again with identical input/settings.
    install_uniform_kv_quant(model, num_bits=2)
    try:
        with torch.no_grad():
            patched_out = model.generate(**inputs, **gen_kwargs)
        patched_ids = patched_out[0].tolist()
        print(f"[smoke_test] patched (2-bit KV quant) generated ids:   {patched_ids}")
    except Exception:
        print("[smoke_test] EXCEPTION during patched generate():")
        traceback.print_exc()
        uninstall_uniform_kv_quant()
        print("[FAIL] exception raised while generating with the patch installed")
        return False

    differs = patched_ids != baseline_ids

    # 3. Uninstall, generate a third time, confirm it matches the original baseline.
    uninstall_uniform_kv_quant()
    with torch.no_grad():
        restored_out = model.generate(**inputs, **gen_kwargs)
    restored_ids = restored_out[0].tolist()
    print(f"[smoke_test] restored (post-uninstall) generated ids:      {restored_ids}")

    reverts_cleanly = restored_ids == baseline_ids

    print(f"[smoke_test] proxy model used: {used_model_name}")
    print(f"[smoke_test] patch altered output relative to baseline: {differs}")
    print(f"[smoke_test] uninstall cleanly restored baseline output: {reverts_cleanly}")

    if differs and reverts_cleanly:
        print(f"[PASS] cache_patch mechanism verified on proxy model {used_model_name}")
        return True
    else:
        if not differs:
            print("[FAIL] patched output did not differ from baseline (patch had no numerical effect)")
        if not reverts_cleanly:
            print("[FAIL] output after uninstall did not match original baseline (patch not cleanly reversible)")
        return False


if __name__ == "__main__":
    ok = run_smoke_test()
    sys.exit(0 if ok else 1)
