"""
Thin orchestrator for the "baseline-uniform-2bit" experiment: loads
google/gemma-2-9b-it once, installs the uniform 2-bit KV-cache fake-quant
patch, and runs both the GSM8K and MMLU evaluation harnesses against it,
reusing the single loaded model/tokenizer (no reloading).

NOTE: this script has NOT been executed in this run because
google/gemma-2-9b-it is a gated Hugging Face model and no HF token with an
accepted license was available in this environment. See code/README.md.
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cache_patch import install_uniform_kv_quant
from eval_gsm8k import run_gsm8k
from eval_mmlu import run_mmlu

EVAL_LOGS_DIR = Path("/workspace/output/eval_logs")

# This baseline uniformly quantizes EVERY cached K/V element to `NUM_BITS`
# bits with no exceptions (unlike an adaptive/mixed-precision scheme that
# would allocate extra bits to a subset of "unstable" tokens). Consequently
# the average KV-cache bit-width is exactly NUM_BITS by construction, not an
# empirical/estimated average over a mixed allocation.
NUM_BITS = 2
AVERAGE_KV_BITS = float(NUM_BITS)


def main():
    parser = argparse.ArgumentParser(description="Uniform-2bit KV quant baseline orchestrator")
    parser.add_argument("--model_name", type=str, default="google/gemma-2-9b-it")
    parser.add_argument("--n_samples", type=int, default=None, help="Passed through to both evals (GSM8K: overall subset size; MMLU: per-subject cap)")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="GSM8K generation length")
    parser.add_argument("--num_bits", type=int, default=NUM_BITS)
    args = parser.parse_args()

    EVAL_LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # Model identity check: confirm the pinned model string before running.
    print(f"Loaded: {args.model_name}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map="cuda"
    )

    install_uniform_kv_quant(model, num_bits=args.num_bits)

    gsm8k_summary = run_gsm8k(
        model, tokenizer, n_samples=args.n_samples, max_new_tokens=args.max_new_tokens
    )
    mmlu_summary = run_mmlu(model, tokenizer, n_samples=args.n_samples)

    average_kv_bits = float(args.num_bits)

    final_summary = {
        "gsm8k_accuracy": gsm8k_summary["gsm8k_accuracy"],
        "mmlu_accuracy": mmlu_summary["mmlu_accuracy"],
        "average_kv_bits": average_kv_bits,
    }

    with open(EVAL_LOGS_DIR / "baseline_summary.json", "w") as f:
        json.dump(final_summary, f, indent=2)

    print(f"[run_baseline] final summary: {final_summary}")


if __name__ == "__main__":
    main()
