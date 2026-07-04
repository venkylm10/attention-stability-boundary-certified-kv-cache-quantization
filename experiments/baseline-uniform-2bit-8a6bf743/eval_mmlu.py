"""
MMLU evaluation harness for the "baseline-uniform-2bit" experiment.

This harness is written to be syntactically correct and logically complete,
but has NOT been executed end-to-end against the pinned `google/gemma-2-9b-it`
model in this run: that model is gated on Hugging Face and no HF token with
an accepted license was available in this environment. See code/README.md
for details.

Standard MMLU protocol: for each test question, build a prompt with the
question + 4 answer choices labeled A-D, preceded by 5 in-subject few-shot
exemplars drawn from the "dev" split. The model's answer is read off as the
argmax over the 4 candidate next-token log-probabilities for " A"/" B"/" C"/
" D" (single forward pass, logits at the last prompt position).

Edge case: some MMLU subjects have fewer than 5 dev rows available for
few-shot exemplars. In that case we use as many dev rows as exist for that
subject (falling back to 0-shot for a subject with literally none) rather
than erroring out or borrowing exemplars from other subjects, since MMLU's
standard protocol is explicitly "in-subject" few-shot.

Tokenization of the candidate letters: transformers tokenizers frequently
split " A" (leading space) into a single token in one tokenizer's vocab but
not another's (e.g. SentencePiece-based tokenizers often prefer a leading
space to be part of the first sub-token, while BPE-based ones may not). We
try both " A"/"A" (and the corresponding B/C/D forms) at load time, pick
whichever of the two variants tokenizes ALL of A/B/C/D as single tokens
(preferring the leading-space form if both do, since the natural completion
position always has a preceding space in this prompt format), and use those
resolved single-token ids as the 4 candidates for every example.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from cache_patch import install_uniform_kv_quant

EVAL_LOGS_DIR = Path("/workspace/output/eval_logs")
MODEL_OUTPUTS_DIR = Path("/workspace/output/model_outputs")

CHOICE_LETTERS = ["A", "B", "C", "D"]
N_FEWSHOT = 5


def resolve_choice_token_ids(tokenizer) -> tuple[list[int], str]:
    """Determine, for this tokenizer, whether " A"/" B"/" C"/" D" (leading
    space) or "A"/"B"/"C"/"D" (no leading space) tokenizes each letter as a
    SINGLE token more consistently, and return the resolved list of 4 token
    ids plus which variant was used (for logging)."""

    def single_token_ids(strings):
        ids = []
        for s in strings:
            toks = tokenizer.encode(s, add_special_tokens=False)
            ids.append(toks[0] if len(toks) == 1 else None)
        return ids

    spaced = single_token_ids([f" {c}" for c in CHOICE_LETTERS])
    bare = single_token_ids(list(CHOICE_LETTERS))

    spaced_ok = sum(x is not None for x in spaced)
    bare_ok = sum(x is not None for x in bare)

    if spaced_ok >= bare_ok and spaced_ok == 4:
        return spaced, "spaced (' A'/' B'/' C'/' D')"
    if bare_ok == 4:
        return bare, "bare ('A'/'B'/'C'/'D')"
    # Neither variant is fully single-token for all 4 letters; prefer
    # whichever has more single-token hits and fall back to the first
    # sub-token id for any letter that split into multiple tokens.
    if spaced_ok >= bare_ok:
        ids = [
            (i if i is not None else tokenizer.encode(f" {c}", add_special_tokens=False)[0])
            for i, c in zip(spaced, CHOICE_LETTERS)
        ]
        return ids, "spaced (' A'/' B'/' C'/' D'), partial single-token coverage"
    ids = [
        (i if i is not None else tokenizer.encode(c, add_special_tokens=False)[0])
        for i, c in zip(bare, CHOICE_LETTERS)
    ]
    return ids, "bare ('A'/'B'/'C'/'D'), partial single-token coverage"


def format_question(question: str, choices: list[str], answer_letter: str | None = None) -> str:
    lines = [question]
    for letter, choice in zip(CHOICE_LETTERS, choices):
        lines.append(f"{letter}. {choice}")
    lines.append("Answer:")
    if answer_letter is not None:
        lines.append(f" {answer_letter}")
    return "\n".join(lines) if answer_letter is None else "\n".join(lines[:-1]) + f"\nAnswer: {answer_letter}"


def build_prompt(subject: str, dev_examples: list[dict], test_example: dict) -> str:
    header = f"The following are multiple choice questions (with answers) about {subject.replace('_', ' ')}.\n\n"
    blocks = []
    for ex in dev_examples:
        answer_letter = CHOICE_LETTERS[ex["answer"]]
        blocks.append(format_question(ex["question"], ex["choices"], answer_letter))
    blocks.append(format_question(test_example["question"], test_example["choices"], answer_letter=None))
    return header + "\n\n".join(blocks)


def run_mmlu(model, tokenizer, n_samples: int | None = None) -> dict:
    """Run the MMLU eval given an already-loaded (and already KV-quant-patched)
    model + tokenizer. `n_samples` is a PER-SUBJECT subsample cap (None = full
    14042-example test set). Returns the summary dict and writes transcripts /
    samples / summary files."""
    EVAL_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    test_ds = load_dataset("cais/mmlu", "all", split="test")
    dev_ds = load_dataset("cais/mmlu", "all", split="dev")

    # Group dev examples by subject for in-subject few-shot exemplars.
    dev_by_subject: dict[str, list[dict]] = {}
    for ex in dev_ds:
        dev_by_subject.setdefault(ex["subject"], []).append(ex)

    # Group test examples by subject, applying the per-subject subsample cap.
    test_by_subject: dict[str, list[dict]] = {}
    for ex in test_ds:
        test_by_subject.setdefault(ex["subject"], []).append(ex)
    if n_samples is not None:
        test_by_subject = {
            subj: rows[: min(n_samples, len(rows))] for subj, rows in test_by_subject.items()
        }

    choice_token_ids, variant_used = resolve_choice_token_ids(tokenizer)
    print(f"[eval_mmlu] resolved answer-letter token variant: {variant_used}, token_ids={choice_token_ids}")

    device = next(model.parameters()).device

    n_correct = 0
    n_total = 0

    transcripts_path = EVAL_LOGS_DIR / "mmlu_transcripts.jsonl"
    samples_path = MODEL_OUTPUTS_DIR / "mmlu_samples.jsonl"

    global_idx = 0
    with open(transcripts_path, "w") as f_transcripts, open(samples_path, "w") as f_samples:
        for subject, test_rows in test_by_subject.items():
            dev_rows = dev_by_subject.get(subject, [])
            # Edge case: fewer than 5 dev rows for this subject -> use all
            # available (possibly 0), rather than borrowing cross-subject.
            fewshot_rows = dev_rows[: min(N_FEWSHOT, len(dev_rows))]
            if len(fewshot_rows) < N_FEWSHOT:
                print(
                    f"[eval_mmlu] subject={subject!r} has only {len(dev_rows)} dev rows "
                    f"(< {N_FEWSHOT}); using {len(fewshot_rows)}-shot in-subject exemplars."
                )

            for ex in test_rows:
                prompt = build_prompt(subject, fewshot_rows, ex)
                inputs = tokenizer(prompt, return_tensors="pt").to(device)

                with torch.no_grad():
                    outputs = model(**inputs)
                last_logits = outputs.logits[0, -1, :]
                candidate_logits = last_logits[choice_token_ids]
                log_probs = F.log_softmax(candidate_logits, dim=-1)
                pred_idx = int(torch.argmax(log_probs).item())
                pred_letter = CHOICE_LETTERS[pred_idx]

                ref_idx = ex["answer"]
                ref_letter = CHOICE_LETTERS[ref_idx]
                correct = pred_idx == ref_idx

                n_total += 1
                n_correct += int(correct)

                record = {
                    "idx": global_idx,
                    "subject": subject,
                    "question": ex["question"],
                    "choices": ex["choices"],
                    "reference_letter": ref_letter,
                    "predicted_letter": pred_letter,
                    "candidate_log_probs": log_probs.tolist(),
                    "correct": correct,
                }
                f_transcripts.write(json.dumps(record) + "\n")
                if global_idx < 20:
                    f_samples.write(json.dumps(record) + "\n")
                global_idx += 1

    accuracy = 100.0 * n_correct / n_total if n_total > 0 else 0.0
    summary = {"mmlu_accuracy": accuracy, "n_examples": n_total}

    with open(EVAL_LOGS_DIR / "mmlu_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"mmlu_accuracy: {accuracy:.2f} ({n_correct}/{n_total})")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MMLU eval harness (uniform-2bit KV quant baseline)")
    parser.add_argument("--n_samples", type=int, default=None, help="Per-subject subsample cap; default None = full 14042-example test set")
    parser.add_argument("--num_bits", type=int, default=2)
    parser.add_argument("--model_name", type=str, default="google/gemma-2-9b-it")
    args = parser.parse_args()

    print(f"Loaded: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    install_uniform_kv_quant(model, num_bits=args.num_bits)

    run_mmlu(model, tokenizer, n_samples=args.n_samples)
