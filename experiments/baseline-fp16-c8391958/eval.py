#!/usr/bin/env python3
"""FP16 baseline evaluation harness for google/gemma-2-9b-it on GSM8K + MMLU.

Intended to run via the EleutherAI lm-evaluation-harness. NOT executed in
this run: google/gemma-2-9b-it is a gated Hugging Face model and no HF_TOKEN
was available in this pod (see /workspace/output/report.md). Left in place,
ready to run once a valid token with an accepted Gemma-2 license is supplied.

Usage:
    export HF_TOKEN=<token>
    python eval.py --task gsm8k --model google/gemma-2-9b-it --dtype float16 \
        --output_dir /workspace/output/gsm8k_generations \
        --log_dir /workspace/output/eval_logs
    python eval.py --task mmlu --model google/gemma-2-9b-it --dtype float16 \
        --output_dir /workspace/output/eval_logs/mmlu \
        --log_dir /workspace/output/eval_logs
"""

import argparse
import json
import os
import time

MODEL_NAME = "google/gemma-2-9b-it"  # pinned by TASK.yaml.subject — do not substitute
AVERAGE_KV_BITS = 16  # FP16, uniform precision, no quantization


def run_task(task: str, model: str, dtype: str, output_dir: str, log_dir: str, seed: int) -> dict:
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    lm = HFLM(pretrained=model, dtype=dtype, device="cuda")

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=[task],
        num_fewshot=8 if task == "gsm8k" else 5,
        batch_size="auto",
        write_out=True,
        log_samples=True,
        random_seed=seed,
    )

    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_path = os.path.join(log_dir, f"{task}_{ts}.json")
    with open(out_path, "w") as f:
        json.dump(results["results"], f, indent=2)

    if task == "gsm8k" and "samples" in results:
        gen_path = os.path.join(output_dir, f"gsm8k_generations_{ts}.jsonl")
        with open(gen_path, "w") as f:
            for sample in results["samples"].get("gsm8k", []):
                f.write(json.dumps(sample) + "\n")

    return results["results"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["gsm8k", "mmlu"])
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.model != MODEL_NAME:
        raise SystemExit(
            f"Refusing to substitute model: TASK.yaml.subject pins '{MODEL_NAME}', got '{args.model}'"
        )

    results = run_task(args.task, args.model, args.dtype, args.output_dir, args.log_dir, args.seed)
    print(json.dumps({"task": args.task, "average_kv_bits": AVERAGE_KV_BITS, "results": results}, indent=2))


if __name__ == "__main__":
    main()
