"""
Standalone, architecture-agnostic uniform N-bit fake-quantization for KV-cache
tensors.

This module has NO dependency on transformers/torch model internals — it just
operates on arbitrary tensors. It is used by cache_patch.py to quantize the
key/value states written into (and read back out of) the KV cache.

Quantization scheme: asymmetric (zero-point) min-max quantization, grouped in
chunks of `group_size` elements along `axis`. This is "fake" quantization:
values are quantized to `num_bits`-bit integer levels and immediately
dequantized back to floating point, so the returned tensor has the same shape
and dtype as the input (no bit-packing is performed — the purpose is to
inject the same numerical error a real low-bit storage format would incur,
without needing an actual packed integer representation).

Default `axis=-1` (head_dim) with `group_size=None` (whole axis is one group)
corresponds to *per-token* quantization for KV tensors shaped
`[batch, heads, seq, head_dim]`: each (batch, head, seq) row gets its own
min/max/scale/zero-point computed over its head_dim values.
"""

import torch


def quantize_dequantize(
    x: torch.Tensor,
    num_bits: int,
    group_size: int | None = None,
    axis: int = -1,
) -> torch.Tensor:
    """
    Asymmetric (zero-point) min-max fake-quantization along `axis`.

    Args:
        x: input tensor of any shape.
        num_bits: number of bits for the quantized integer levels. Levels are
            clamped to the integer range [0, 2**num_bits - 1].
        group_size: number of elements per quantization group along `axis`.
            If None, the entire axis length is treated as a single group
            (per-token quantization for a `[..., seq, head_dim]` KV tensor
            with axis=-1).
        axis: the axis along which groups are formed (default -1, i.e. the
            last/head_dim axis for KV tensors).

    Returns:
        A tensor of the same shape and dtype as `x`, with values replaced by
        their quantize-then-dequantize reconstruction.
    """
    orig_dtype = x.dtype
    orig_shape = x.shape

    # Work in float32 for numerically stable min/max/scale computation,
    # regardless of the input's dtype (e.g. bfloat16 KV states).
    xf = x.to(torch.float32)

    # Move `axis` to be the last dimension so grouping logic is uniform.
    axis = axis % xf.dim()
    if axis != xf.dim() - 1:
        xf = xf.movedim(axis, -1)
    axis_len = xf.shape[-1]

    gs = group_size if group_size is not None else axis_len
    if gs <= 0:
        raise ValueError(f"group_size must be a positive integer, got {gs}")

    # Pad the last dimension up to a multiple of the group size so we can
    # reshape into groups. Padding uses replication of the last element so
    # the pad values don't distort the group's min/max (they get overwritten
    # back to themselves after dequant and are sliced off before returning).
    n_groups = -(-axis_len // gs)  # ceil division
    padded_len = n_groups * gs
    pad_amount = padded_len - axis_len
    if pad_amount > 0:
        pad_vals = xf[..., -1:].expand(*xf.shape[:-1], pad_amount)
        xf_padded = torch.cat([xf, pad_vals], dim=-1)
    else:
        xf_padded = xf

    grouped = xf_padded.reshape(*xf_padded.shape[:-1], n_groups, gs)

    g_min = grouped.amin(dim=-1, keepdim=True)
    g_max = grouped.amax(dim=-1, keepdim=True)

    qmax = float(2**num_bits - 1)

    value_range = g_max - g_min
    # Degenerate case: a group's max == min (zero range) -> avoid div-by-zero.
    # In that case every element in the group is identical, so scale is
    # irrelevant; set scale to 1.0 and the quantized value will simply
    # reconstruct back to g_min (== g_max == every element in the group).
    zero_range_mask = value_range <= 0
    safe_range = torch.where(
        zero_range_mask, torch.ones_like(value_range), value_range
    )
    scale = safe_range / qmax  # step size

    # Asymmetric quant: zero_point chosen so that g_min maps to integer 0.
    q = torch.round((grouped - g_min) / scale)
    q = torch.clamp(q, 0, qmax)

    dequant = q * scale + g_min
    # For zero-range groups, force exact reconstruction of the constant value.
    dequant = torch.where(zero_range_mask, g_min, dequant)

    # Undo the group reshape + padding.
    flat = dequant.reshape(*dequant.shape[:-2], padded_len)
    if pad_amount > 0:
        flat = flat[..., :axis_len]

    # Move axis back to its original position.
    if axis != len(orig_shape) - 1:
        flat = flat.movedim(-1, axis)

    return flat.to(orig_dtype).reshape(orig_shape)


if __name__ == "__main__":
    torch.manual_seed(0)

    # ------------------------------------------------------------------
    # (a) Hand-checkable 3-token x 4-dim key tensor, 2-bit quantization.
    # ------------------------------------------------------------------
    # Shape [batch=1, heads=1, seq=3, head_dim=4]
    x = torch.tensor(
        [[[
            [0.0, 1.0, 2.0, 3.0],
            [-1.0, 0.0, 1.0, 2.0],
            [10.0, 10.0, 10.0, 10.0],  # zero-range group (degenerate case)
        ]]],
        dtype=torch.float32,
    )
    num_bits_a = 2
    out = quantize_dequantize(x, num_bits=num_bits_a, group_size=None, axis=-1)
    assert out.shape == x.shape
    assert out.dtype == x.dtype

    qmax_a = 2**num_bits_a - 1
    for token_idx in range(x.shape[-2]):
        row = x[0, 0, token_idx]
        recon = out[0, 0, token_idx]
        g_min = row.min().item()
        g_max = row.max().item()
        rng = g_max - g_min
        step = rng / qmax_a if rng > 0 else 0.0
        max_err = (row - recon).abs().max().item()
        if step > 0:
            assert max_err <= step / 2 + 1e-5, (
                f"token {token_idx}: max round-trip err {max_err} exceeds step/2 {step / 2}"
            )
        else:
            # Degenerate zero-range group: reconstruction must be exact.
            assert max_err <= 1e-6, (
                f"token {token_idx}: degenerate group did not reconstruct exactly, err={max_err}"
            )
    print("[a] hand-checkable round-trip test passed:")
    print("    input:\n", x)
    print("    dequantized:\n", out)

    # ------------------------------------------------------------------
    # (b) Randomized tensor at 2/4/8 bits: mean abs error strictly decreases.
    # ------------------------------------------------------------------
    x_rand = torch.randn(2, 8, 37, 64, dtype=torch.float32)  # [batch, heads, seq, head_dim]
    errs = {}
    for nb in (2, 4, 8):
        out_rand = quantize_dequantize(x_rand, num_bits=nb, group_size=None, axis=-1)
        err = (x_rand - out_rand).abs().mean().item()
        errs[nb] = err

    print(f"[b] mean abs error by bit-width: {errs}")
    assert errs[8] < errs[4] < errs[2], (
        f"expected err(8) < err(4) < err(2), got {errs}"
    )

    # Also sanity-check a non-default group_size / axis combination runs and
    # shapes are preserved (not part of the required assertions above, but a
    # useful extra smoke check).
    out_grouped = quantize_dequantize(x_rand, num_bits=4, group_size=16, axis=-1)
    assert out_grouped.shape == x_rand.shape

    print("[PASS]")
