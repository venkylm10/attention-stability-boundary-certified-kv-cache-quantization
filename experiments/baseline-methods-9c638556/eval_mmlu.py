"""MMLU evaluation: 5-shot, multiple-choice scored via next-token log-likelihood over A/B/C/D.

Log-likelihood scoring (rather than free-form generation) is the standard MMLU
protocol (Hendrycks et al.) and is far cheaper per example than CoT generation,
which matters given the wall-clock budget for 5 methods x N examples.
"""

from __future__ import annotations

import json
import random
import time

import torch
from datasets import load_dataset

from quant_methods import QuantConfig, QuantizedCache

LETTERS = ["A", "B", "C", "D"]


def _format_question(q: str, choices: list[str], answer: int | None = None) -> str:
    lines = [q]
    for letter, choice in zip(LETTERS, choices):
        lines.append(f"{letter}. {choice}")
    lines.append(f"Answer:{(' ' + LETTERS[answer]) if answer is not None else ''}")
    return "\n".join(lines)


def _build_fewshot_prefix(dev_examples) -> str:
    blocks = []
    for ex in dev_examples:
        blocks.append(_format_question(ex["question"], ex["choices"], ex["answer"]))
    return "\n\n".join(blocks) + "\n\n"


def run_mmlu(model, tokenizer, method: str, n_examples: int, log_path: str,
             generations_path: str, n_fewshot: int = 5, seed: int = 42) -> dict:
    test_ds = load_dataset("cais/mmlu", "all", split="test")
    dev_ds = load_dataset("cais/mmlu", "all", split="dev")

    rng = random.Random(seed)
    idxs = list(range(len(test_ds)))
    rng.shuffle(idxs)
    idxs = idxs[:n_examples]

    letter_ids = [tokenizer.encode(f" {l}", add_special_tokens=False)[-1] for l in LETTERS]

    correct = 0
    total = 0
    log_f = open(log_path, "a")
    gen_f = open(generations_path, "w")
    t_start = time.time()
    for i, idx in enumerate(idxs):
        ex = test_ds[idx]
        subject = ex["subject"]
        subj_dev = [d for d in dev_ds if d["subject"] == subject][:n_fewshot]
        prefix = _build_fewshot_prefix(subj_dev) if subj_dev else ""
        prompt = prefix + _format_question(ex["question"], ex["choices"])

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        cache = QuantizedCache(QuantConfig(method=method, seed=seed)) if method != "fp16" else None
        with torch.no_grad():
            out = model(**inputs, past_key_values=cache, use_cache=True)
        logits = out.logits[0, -1, :]
        choice_logits = logits[letter_ids]
        pred_letter = LETTERS[int(torch.argmax(choice_logits))]
        gold_letter = LETTERS[ex["answer"]]
        is_correct = pred_letter == gold_letter
        correct += int(is_correct)
        total += 1
        gen_f.write(json.dumps({"idx": int(idx), "subject": subject, "pred": pred_letter,
                                 "gold": gold_letter, "correct": is_correct}) + "\n")
        if i % 25 == 0:
            log_f.write(f"[mmlu method={method}] step {i}/{len(idxs)} running_acc="
                        f"{correct / max(total, 1):.4f}\n")
            log_f.flush()
    elapsed = time.time() - t_start
    log_f.write(f"=== mmlu method={method} done | n={total} accuracy={100 * correct / total:.2f}% "
                f"elapsed={elapsed:.1f}s ===\n")
    log_f.close()
    gen_f.close()
    return {"accuracy_pct": 100.0 * correct / total, "n": total, "elapsed_s": elapsed}
