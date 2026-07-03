"""Theoretical KV-cache memory footprint per method, given model architecture.

kv_cache_size is reported in MB and is a deterministic function of the model's
architecture (num layers, num KV heads, head dim), the sequence length used for
the size measurement, and the method's effective bits/element (see
`quant_methods.effective_bits`). This mirrors how each baseline paper reports
compression ratio, and avoids conflating it with actual measured GPU memory
(which includes activations, KV cache padding, and allocator fragmentation
that vary run to run).
"""

from __future__ import annotations

from quant_methods import effective_bits

# Gemma-2-9B-it architecture (from the model config; see README.md for the
# exact values once the pinned checkpoint can be loaded and confirmed).
GEMMA2_9B_NUM_LAYERS = 42
GEMMA2_9B_NUM_KV_HEADS = 8
GEMMA2_9B_HEAD_DIM = 256


def kv_cache_size_mb(method: str, seq_len: int = 4096, batch_size: int = 1,
                      num_layers: int = GEMMA2_9B_NUM_LAYERS,
                      num_kv_heads: int = GEMMA2_9B_NUM_KV_HEADS,
                      head_dim: int = GEMMA2_9B_HEAD_DIM, **method_kwargs) -> float:
    """Total K+V cache size in MB for a given method at a fixed sequence length."""
    bits = effective_bits(method, head_dim=head_dim, seq_len=seq_len, **method_kwargs)
    elements = 2 * batch_size * num_layers * num_kv_heads * seq_len * head_dim  # K and V
    total_bits = elements * bits
    return total_bits / 8 / (1024 ** 2)


if __name__ == "__main__":
    for method in ["fp16", "uniform2bit", "gear", "rotatekv", "kitty"]:
        print(f"{method:>12}: {kv_cache_size_mb(method):.2f} MB @ seq_len=4096")
