"""Main entry point for the baseline-methods experiment.

There is no gradient training in this experiment — "baseline-methods" means
establishing FP16 and standard KV-cache-quantization baseline *evaluation*
numbers (accuracy/latency/cache-size) for google/gemma-2-9b-it, to anchor
comparisons for the sibling ASB-KVQ experiments. This script is the single
entry point named `train.py` per the project's ML output-layout convention;
`eval.py` in this directory is a thin alias to the same pipeline.

Usage:
    python train.py --n_gsm8k 200 --n_mmlu 500 --out_dir /workspace/output

Requires a valid HF_TOKEN (or `huggingface-cli login`) with the Gemma-2
license accepted at https://huggingface.co/google/gemma-2-9b-it — this
experiment's run on 2026-07-03 was BLOCKED because no such token was present
in the pod environment. See /workspace/output/report.md for details. This
code is otherwise complete and ready to run once access is available.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_gsm8k import run_gsm8k
from eval_mmlu import run_mmlu
from latency_bench import measure_decoding_latency
from kv_cache_size import kv_cache_size_mb, GEMMA2_9B_NUM_LAYERS, GEMMA2_9B_NUM_KV_HEADS, GEMMA2_9B_HEAD_DIM

MODEL_NAME = "google/gemma-2-9b-it"  # pinned — do not substitute
METHODS = ["fp16", "uniform2bit", "gear", "rotatekv", "kitty"]
SEED = 42


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gsm8k", type=int, default=200)
    parser.add_argument("--n_mmlu", type=int, default=500)
    parser.add_argument("--out_dir", type=str, default="/workspace/output")
    parser.add_argument("--methods", type=str, default=",".join(METHODS))
    args = parser.parse_args()

    torch.manual_seed(SEED)

    logs_dir = os.path.join(args.out_dir, "baseline_logs")
    gens_dir = os.path.join(args.out_dir, "baseline_generations")
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(gens_dir, exist_ok=True)
    train_log_path = os.path.join(args.out_dir, "train.log")

    def log(msg: str):
        print(msg)
        with open(train_log_path, "a") as f:
            f.write(msg + "\n")

    log(f"Loading pinned model: {MODEL_NAME}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        ).to("cuda")
        model.eval()
    except Exception as e:
        log(f"FATAL: could not load {MODEL_NAME}: {e}")
        log("STOP — per project policy, do not substitute a different model. "
            "Set validation_status=failed and report this blocker.")
        sys.exit(1)

    log(f"Loaded: {MODEL_NAME}")
    cfg = model.config
    log(f"Config: num_layers={cfg.num_hidden_layers} num_kv_heads={cfg.num_key_value_heads} "
        f"head_dim={cfg.head_dim if hasattr(cfg, 'head_dim') else cfg.hidden_size // cfg.num_attention_heads}")

    all_results = {}
    for method in args.methods.split(","):
        log(f"--- method={method} ---")
        gsm8k_log = os.path.join(logs_dir, f"{method}_gsm8k.log")
        gsm8k_gen = os.path.join(gens_dir, f"{method}_gsm8k.jsonl")
        gsm8k_res = run_gsm8k(model, tokenizer, method, args.n_gsm8k, gsm8k_log, gsm8k_gen, seed=SEED)
        log(f"[gsm8k] method={method} accuracy={gsm8k_res['accuracy_pct']:.2f}% "
            f"n={gsm8k_res['n']} elapsed={gsm8k_res['elapsed_s']:.1f}s")

        mmlu_log = os.path.join(logs_dir, f"{method}_mmlu.log")
        mmlu_gen = os.path.join(gens_dir, f"{method}_mmlu.jsonl")
        mmlu_res = run_mmlu(model, tokenizer, method, args.n_mmlu, mmlu_log, mmlu_gen, seed=SEED)
        log(f"[mmlu] method={method} accuracy={mmlu_res['accuracy_pct']:.2f}% "
            f"n={mmlu_res['n']} elapsed={mmlu_res['elapsed_s']:.1f}s")

        lat_res = measure_decoding_latency(model, tokenizer, method, seed=SEED)
        log(f"[latency] method={method} ms_per_token={lat_res['ms_per_token']:.3f}")

        cache_mb = kv_cache_size_mb(method, num_layers=cfg.num_hidden_layers,
                                     num_kv_heads=cfg.num_key_value_heads,
                                     head_dim=getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads))
        log(f"[kv_cache_size] method={method} size_mb={cache_mb:.2f}")

        all_results[method] = {
            "gsm8k_accuracy": gsm8k_res["accuracy_pct"],
            "mmlu_accuracy": mmlu_res["accuracy_pct"],
            "decoding_latency_ms_per_token": lat_res["ms_per_token"],
            "kv_cache_size_mb": cache_mb,
        }

    with open(os.path.join(args.out_dir, "baseline_raw_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    log("Done. Raw per-method results written to baseline_raw_results.json — "
        "assemble the enriched results.json manifest from this file.")


if __name__ == "__main__":
    main()
