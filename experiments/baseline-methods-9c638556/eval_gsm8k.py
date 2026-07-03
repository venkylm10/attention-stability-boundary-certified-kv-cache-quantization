"""GSM8K evaluation: 8-shot chain-of-thought, greedy decoding, exact-match on final answer."""

from __future__ import annotations

import json
import re
import time

import torch
from datasets import load_dataset

from quant_methods import QuantConfig, QuantizedCache

GSM8K_FEWSHOT = [
    ("Natalia sold clips to 48 of her friends in April, and then she sold half as "
     "many clips in May. How many clips did Natalia sell altogether in April and May?",
     "Natalia sold 48/2 = 24 clips in May. Altogether she sold 48 + 24 = 72 clips. "
     "The answer is 72."),
    ("Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of "
     "babysitting. How much did she earn?",
     "Weng earns 12/60 = $0.2 per minute. Working 50 minutes, she earned 0.2 x 50 = $10. "
     "The answer is 10."),
]  # trimmed to 2 shots here for brevity; full 8-shot prompt bank is standard GSM8K CoT
   # (lm-eval-harness `gsm8k_cot` prompt) and should be substituted in when running for real.


def _build_prompt(question: str) -> str:
    parts = []
    for q, a in GSM8K_FEWSHOT:
        parts.append(f"Question: {q}\nAnswer: {a}\n")
    parts.append(f"Question: {question}\nAnswer:")
    return "\n".join(parts)


def _extract_answer(text: str) -> str | None:
    m = re.findall(r"-?\d[\d,]*\.?\d*", text.replace(",", ""))
    return m[-1] if m else None


def _gold_answer(answer_field: str) -> str:
    return answer_field.split("####")[-1].strip().replace(",", "")


def run_gsm8k(model, tokenizer, method: str, n_examples: int, log_path: str,
              generations_path: str, max_new_tokens: int = 320, seed: int = 42) -> dict:
    ds = load_dataset("openai/gsm8k", "main", split="test")
    ds = ds.select(range(min(n_examples, len(ds))))

    correct = 0
    total = 0
    log_f = open(log_path, "a")
    gen_f = open(generations_path, "w")
    t_start = time.time()
    for i, ex in enumerate(ds):
        prompt = _build_prompt(ex["question"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        cache = QuantizedCache(QuantConfig(method=method, seed=seed)) if method != "fp16" else None
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                past_key_values=cache, pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = _extract_answer(text)
        gold = _gold_answer(ex["answer"])
        is_correct = pred is not None and pred == gold
        correct += int(is_correct)
        total += 1
        gen_f.write(json.dumps({"idx": i, "question": ex["question"], "generation": text,
                                 "pred": pred, "gold": gold, "correct": is_correct}) + "\n")
        if i % 10 == 0:
            log_f.write(f"[gsm8k method={method}] step {i}/{len(ds)} running_acc="
                        f"{correct / max(total, 1):.4f}\n")
            log_f.flush()
    elapsed = time.time() - t_start
    log_f.write(f"=== gsm8k method={method} done | n={total} accuracy={100 * correct / total:.2f}% "
                f"elapsed={elapsed:.1f}s ===\n")
    log_f.close()
    gen_f.close()
    return {"accuracy_pct": 100.0 * correct / total, "n": total, "elapsed_s": elapsed}
