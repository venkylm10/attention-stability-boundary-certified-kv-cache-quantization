"""Uninstrumented decoding-latency benchmark.

Deliberately separate from eval_gsm8k.py / eval_mmlu.py: the methodology
review flagged that instrumenting the generation loop to verify the
theoretical attention bound (needed for the ASB-KVQ sibling experiments)
would artificially inflate latency and invalidate a "zero overhead" claim.
For this baseline-methods experiment there is no bound-checking to begin
with, but we still keep latency measurement in its own uninstrumented path
so the methodology is consistent with the sibling experiments and the number
reported here is a clean, comparable decoding_latency baseline.

Reports ms/token, averaged over `n_repeats` runs after `n_warmup` warmup runs,
batch size 1, fixed prompt length, greedy decoding.
"""

from __future__ import annotations

import time

import torch

from quant_methods import QuantConfig, QuantizedCache


def measure_decoding_latency(model, tokenizer, method: str, prompt: str = "The quick brown fox",
                              max_new_tokens: int = 128, n_warmup: int = 2, n_repeats: int = 5,
                              seed: int = 42) -> dict:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    def _one_run():
        cache = QuantizedCache(QuantConfig(method=method, seed=seed)) if method != "fp16" else None
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                            past_key_values=cache, pad_token_id=tokenizer.eos_token_id)
        torch.cuda.synchronize()
        return time.time() - t0

    for _ in range(n_warmup):
        _one_run()

    times = [_one_run() for _ in range(n_repeats)]
    ms_per_token = [1000.0 * t / max_new_tokens for t in times]
    mean_ms_per_token = sum(ms_per_token) / len(ms_per_token)
    return {"ms_per_token": mean_ms_per_token, "raw_run_times_s": times, "n_repeats": n_repeats}
