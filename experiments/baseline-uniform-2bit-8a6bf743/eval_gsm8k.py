"""
GSM8K evaluation harness for the "baseline-uniform-2bit" experiment.

This harness is written to be syntactically correct and logically complete,
but has NOT been executed end-to-end against the pinned `google/gemma-2-9b-it`
model in this run: that model is gated on Hugging Face and no HF token with
an accepted license was available in this environment. See code/README.md
for details.

Few-shot (not zero-shot) prompting is used: 4 hand-written GSM8K exemplars
(question + explicit chain-of-thought + a final "#### <answer>" line) are
prepended before the actual test question. Few-shot exemplars make the model
reliably emit the "#### <number>" terminator we need for robust answer
extraction — zero-shot generations are much more likely to omit or vary the
final-answer format, which would depress measured accuracy for reasons
unrelated to the quantization method under test.
"""

import argparse
import json
import os
import re
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from cache_patch import install_uniform_kv_quant

EVAL_LOGS_DIR = Path("/workspace/output/eval_logs")
MODEL_OUTPUTS_DIR = Path("/workspace/output/model_outputs")

ANSWER_RE = re.compile(r"####\s*(-?[0-9][0-9,.]*)")
NUMBER_RE = re.compile(r"-?[0-9][0-9,.]*")

# 4 hand-written few-shot exemplars, each with explicit chain-of-thought
# reasoning and a terminal "#### <answer>" line matching GSM8K's own
# reference-answer format.
FEWSHOT_EXEMPLARS = [
    {
        "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
        "cot": "In April, Natalia sold 48 clips.\nIn May, she sold half as many, so she sold 48 / 2 = 24 clips.\nAltogether she sold 48 + 24 = 72 clips.\n#### 72",
    },
    {
        "question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
        "cot": "Weng earns $12 per 60 minutes, so per minute she earns 12 / 60 = $0.2.\nFor 50 minutes she earned 50 * 0.2 = $10.\n#### 10",
    },
    {
        "question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?",
        "cot": "Betty has half of $100, which is 100 / 2 = $50.\nHer parents give her $15.\nHer grandparents give twice as much as her parents, which is 15 * 2 = $30.\nIn total Betty now has 50 + 15 + 30 = $95.\nShe still needs 100 - 95 = $5.\n#### 5",
    },
    {
        "question": "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?",
        "cot": "Each time, James writes 3 pages to 2 friends, so 3 * 2 = 6 pages.\nHe does this twice a week, so 6 * 2 = 12 pages per week.\nIn a year there are 52 weeks, so he writes 12 * 52 = 624 pages a year.\n#### 624",
    },
]

FINAL_INSTRUCTION = (
    "Solve the following grade-school math problem. Show your step-by-step "
    "reasoning, then give the final answer on its own line in the exact "
    "format '#### <number>'."
)


def build_fewshot_messages(question: str) -> list[dict]:
    """Build a chat-template-ready list of {role, content} messages: a
    system-style instruction, the 4 few-shot exemplars as user/assistant
    turns, then the real question as the final user turn."""
    messages = [{"role": "user", "content": FINAL_INSTRUCTION}]
    # Fold the instruction into the first exemplar's user turn so models
    # without a "system" role (e.g. Gemma family) still see it up front.
    first = True
    for ex in FEWSHOT_EXEMPLARS:
        prefix = f"{FINAL_INSTRUCTION}\n\n" if first else ""
        messages.append({"role": "user", "content": f"{prefix}Question: {ex['question']}"})
        messages.append({"role": "assistant", "content": ex["cot"]})
        first = False
    messages.append({"role": "user", "content": f"Question: {question}"})
    return messages


def extract_answer(text: str) -> str | None:
    """Extract the final numeric answer: prefer an explicit '#### <num>'
    line; fall back to the last number appearing anywhere in the text."""
    m = ANSWER_RE.search(text)
    if m:
        return m.group(1)
    nums = NUMBER_RE.findall(text)
    if nums:
        return nums[-1]
    return None


def normalize_number(s: str | None) -> float | None:
    if s is None:
        return None
    s = s.strip().replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def is_correct(pred: str | None, ref: str | None) -> bool:
    pred_f = normalize_number(pred)
    ref_f = normalize_number(ref)
    if pred_f is None or ref_f is None:
        return False
    return abs(pred_f - ref_f) < 1e-4


def run_gsm8k(model, tokenizer, n_samples: int | None = None, max_new_tokens: int = 256) -> dict:
    """Run the GSM8K eval given an already-loaded (and already KV-quant-patched)
    model + tokenizer. Returns the summary dict and also writes transcripts /
    samples / summary files under /workspace/output/eval_logs and
    /workspace/output/model_outputs."""
    EVAL_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("openai/gsm8k", "main", split="test")
    if n_samples is not None:
        dataset = dataset.select(range(min(n_samples, len(dataset))))

    device = next(model.parameters()).device

    n_correct = 0
    n_total = 0

    transcripts_path = EVAL_LOGS_DIR / "gsm8k_transcripts.jsonl"
    samples_path = MODEL_OUTPUTS_DIR / "gsm8k_samples.jsonl"

    with open(transcripts_path, "w") as f_transcripts, open(samples_path, "w") as f_samples:
        for idx, example in enumerate(dataset):
            question = example["question"]
            reference_full = example["answer"]
            reference_answer = extract_answer(reference_full)

            messages = build_fewshot_messages(question)
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                out_ids = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            gen_ids = out_ids[0][inputs["input_ids"].shape[-1]:]
            generated_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

            predicted_answer = extract_answer(generated_text)
            correct = is_correct(predicted_answer, reference_answer)

            n_total += 1
            n_correct += int(correct)

            record = {
                "idx": idx,
                "question": question,
                "reference_answer_raw": reference_full,
                "reference_answer": reference_answer,
                "generated_text": generated_text,
                "predicted_answer": predicted_answer,
                "correct": correct,
            }
            f_transcripts.write(json.dumps(record) + "\n")
            if idx < 20:
                f_samples.write(json.dumps(record) + "\n")

    accuracy = 100.0 * n_correct / n_total if n_total > 0 else 0.0
    summary = {"gsm8k_accuracy": accuracy, "n_examples": n_total}

    with open(EVAL_LOGS_DIR / "gsm8k_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"gsm8k_accuracy: {accuracy:.2f} ({n_correct}/{n_total})")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GSM8K eval harness (uniform-2bit KV quant baseline)")
    parser.add_argument("--n_samples", type=int, default=None, help="Subset size; default None = full 1319-example test split")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--num_bits", type=int, default=2)
    parser.add_argument("--model_name", type=str, default="google/gemma-2-9b-it")
    args = parser.parse_args()

    print(f"Loaded: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    install_uniform_kv_quant(model, num_bits=args.num_bits)

    run_gsm8k(model, tokenizer, n_samples=args.n_samples, max_new_tokens=args.max_new_tokens)
