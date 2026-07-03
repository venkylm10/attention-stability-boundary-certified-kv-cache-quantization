"""KV-cache quantization baselines for ASB-KVQ comparisons.

Implements a common interface — a `transformers.cache_utils.Cache` subclass
that fake-quantizes key/value states at write time and dequantizes them for
attention compute — so accuracy degradation from each scheme can be measured
without needing real low-bit GEMM kernels. Actual memory savings are computed
separately (see `kv_cache_size.py`) from each method's declared bit-width,
since the fake-quantization here keeps tensors in the compute dtype.

Methods implemented (baselines only — NOT the ASB-KVQ method under test,
which is implemented/evaluated in the sibling `main-asb-kvq` experiment):

- `FP16Cache`       — unquantized reference (passthrough).
- `Uniform2BitCache`— per-token asymmetric min/max round-to-nearest 2-bit.
- `GearCache`       — 2-bit quantization + low-rank residual correction +
                       sparse outlier correction (GEAR, arXiv:2403.05527).
- `RotateKVCache`   — random orthogonal (Hadamard-style) rotation before
                       quantization to spread outlier energy across channels,
                       then uniform 2-bit, then rotate back (RotateKV-style).
- `KittyCache`      — dynamic channel-wise precision boost: per-channel
                       variance ranks channels, top-p% get 4-bit, rest 2-bit
                       (Kitty, arXiv:2511.18643).

These are faithful reimplementations of each method's core algorithmic idea,
not line-for-line ports of the reference repos (which depend on custom
CUDA/Triton kernels and pinned library trees not compatible with this pod's
image). This is noted explicitly in report.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from transformers.cache_utils import DynamicCache


def _asymmetric_quantize(x: torch.Tensor, bits: int, dim: int = -1) -> torch.Tensor:
    """Round-to-nearest asymmetric quantization, fake-quantized (dequantized in place).

    Quantizes independently along `dim` (per-token for a [.., seq, head_dim] tensor
    with dim=-1 quantizing per-channel-group across head_dim, matching common
    per-token KV quantization granularity).
    """
    qmax = 2 ** bits - 1
    x_min = x.amin(dim=dim, keepdim=True)
    x_max = x.amax(dim=dim, keepdim=True)
    scale = (x_max - x_min).clamp(min=1e-8) / qmax
    x_q = torch.clamp(torch.round((x - x_min) / scale), 0, qmax)
    return x_q * scale + x_min


def _hadamard_like_rotation(dim: int, device, dtype, generator: torch.Generator) -> torch.Tensor:
    """Random orthogonal rotation matrix (QR of a Gaussian) as a Hadamard-transform stand-in.

    RotateKV uses a fixed Hadamard matrix for its FWHT; we don't have a fast
    Walsh-Hadamard transform kernel available in this environment, so we use a
    random orthogonal matrix, which has the same outlier-decorrelation effect
    (rotation invariance of the L2 quantization error bound) at O(d^2) cost
    instead of O(d log d). Noted as a deviation in report.md.
    """
    g = torch.randn(dim, dim, device=device, dtype=torch.float32, generator=generator)
    q, _ = torch.linalg.qr(g)
    return q.to(dtype)


@dataclass
class QuantConfig:
    method: str  # "fp16" | "uniform2bit" | "gear" | "rotatekv" | "kitty"
    bits: int = 2
    gear_rank: int = 2               # low-rank correction rank
    gear_outlier_frac: float = 0.01  # fraction of largest-residual entries kept in fp16
    kitty_high_bits: int = 4
    kitty_high_frac: float = 0.1     # fraction of channels boosted to kitty_high_bits
    seed: int = 42


class QuantizedCache(DynamicCache):
    """DynamicCache subclass that fake-quantizes K/V per QuantConfig at write time."""

    def __init__(self, config: QuantConfig):
        super().__init__()
        self.qcfg = config
        self._rotation: torch.Tensor | None = None
        self._kitty_channel_rank: dict[int, torch.Tensor] = {}
        self._gen = torch.Generator(device="cpu").manual_seed(config.seed)

    def _quantize(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        cfg = self.qcfg
        if cfg.method == "fp16":
            return x

        if cfg.method == "uniform2bit":
            return _asymmetric_quantize(x, cfg.bits, dim=-1)

        if cfg.method == "gear":
            x_q = _asymmetric_quantize(x, cfg.bits, dim=-1)
            residual = x - x_q
            # Low-rank correction via SVD on the last two dims (seq, head_dim),
            # applied per (batch, head).
            b, h, s, d = x.shape
            r = min(cfg.gear_rank, s, d)
            flat = residual.reshape(b * h, s, d)
            try:
                u, sv, vt = torch.linalg.svd(flat.float(), full_matrices=False)
                low_rank = (u[:, :, :r] * sv[:, None, :r]) @ vt[:, :r, :]
                low_rank = low_rank.reshape(b, h, s, d).to(x.dtype)
            except RuntimeError:
                low_rank = torch.zeros_like(residual)
            corrected = x_q + low_rank
            # Sparse outlier correction: restore the top-k largest remaining residuals exactly.
            remaining = x - corrected
            k = max(1, int(cfg.gear_outlier_frac * remaining.numel() / (b * h)))
            flat_remaining = remaining.reshape(b * h, -1)
            topk = torch.topk(flat_remaining.abs(), k=min(k, flat_remaining.shape[-1]), dim=-1)
            out = corrected.reshape(b * h, -1).clone()
            out.scatter_(-1, topk.indices, flat_remaining.gather(-1, topk.indices) + out.gather(-1, topk.indices))
            return out.reshape(b, h, s, d)

        if cfg.method == "rotatekv":
            d = x.shape[-1]
            if self._rotation is None or self._rotation.shape[0] != d:
                self._rotation = _hadamard_like_rotation(d, x.device, x.dtype, self._gen)
            rot = self._rotation
            x_rot = x @ rot
            x_rot_q = _asymmetric_quantize(x_rot, cfg.bits, dim=-1)
            return x_rot_q @ rot.T

        if cfg.method == "kitty":
            d = x.shape[-1]
            if layer_idx not in self._kitty_channel_rank:
                # Rank channels by variance across (batch, head, seq) on this first call
                # for the layer; precision boost mask is then fixed for the sequence
                # (channel importance is stable across tokens within a layer/head in
                # practice, consistent with Kitty's channel-wise precision assignment).
                var = x.float().var(dim=(0, 1, 2))
                n_high = max(1, int(cfg.kitty_high_frac * d))
                high_idx = torch.topk(var, k=n_high).indices
                mask = torch.zeros(d, dtype=torch.bool, device=x.device)
                mask[high_idx] = True
                self._kitty_channel_rank[layer_idx] = mask
            mask = self._kitty_channel_rank[layer_idx]
            out = torch.empty_like(x)
            out[..., mask] = _asymmetric_quantize(x[..., mask], cfg.kitty_high_bits, dim=-1)
            out[..., ~mask] = _asymmetric_quantize(x[..., ~mask], cfg.bits, dim=-1)
            return out

        raise ValueError(f"unknown quant method {cfg.method!r}")

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        key_states = self._quantize(key_states, layer_idx)
        value_states = self._quantize(value_states, layer_idx)
        return super().update(key_states, value_states, layer_idx, cache_kwargs)


def effective_bits(method: str, kitty_high_bits: int = 4, kitty_high_frac: float = 0.1,
                    gear_rank: int = 2, gear_outlier_frac: float = 0.01, bits: int = 2,
                    head_dim: int = 256, seq_len: int = 1024) -> float:
    """Average bits/element actually needed to store the K or V cache under each method.

    Used by kv_cache_size.py. GEAR and the sparse-outlier/low-rank side info are
    charged their real storage cost (outliers stored as (index, fp16 value) pairs;
    low-rank factors stored in fp32) amortized per element.
    """
    if method == "fp16":
        return 16.0
    if method == "uniform2bit":
        return float(bits)
    if method == "rotatekv":
        # + rotation matrix amortized cost is O(head_dim) shared across the whole
        # sequence, negligible per-element for seq_len >> 1.
        return float(bits) + (32.0 * head_dim) / max(seq_len, 1)
    if method == "kitty":
        return kitty_high_frac * kitty_high_bits + (1 - kitty_high_frac) * bits
    if method == "gear":
        outlier_bits_per_elem = gear_outlier_frac * (16.0 + math.log2(max(seq_len * head_dim, 2)))
        low_rank_bits_per_elem = (2 * gear_rank * 32.0) / max(head_dim, 1)
        return float(bits) + outlier_bits_per_elem + low_rank_bits_per_elem
    raise ValueError(f"unknown method {method!r}")
