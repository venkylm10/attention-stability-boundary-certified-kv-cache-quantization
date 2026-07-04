# baseline-fp16 — Gemma-2-9b-it, FP16 KV cache (GSM8K + MMLU)

## Status: BLOCKED — not executed

This experiment is meant to establish the FP16 upper-bound accuracy baseline
for the ASB-KVQ project (`google/gemma-2-9b-it` on GSM8K and MMLU, tracking
`average_kv_bits`). It could **not** be run: `google/gemma-2-9b-it` is a
gated Hugging Face repository, and no `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN`
is present in this pod's environment (verified: env has only `GITHUB_TOKEN`
and `WANDB_API_KEY`; a direct `hf_hub_download` probe for `config.json`
returned `401 GatedRepoError`). Per the platform's subject-pinning rule, we
did not substitute a different (ungated) model. See `/workspace/output/report.md`
for the full explanation.

`eval.py` below is a complete, ready-to-run harness invocation for this
baseline. It has NOT been executed against the real model. Once a token
with accepted Gemma-2 license terms is supplied as `HF_TOKEN`, running it
end to end should reproduce the intended baseline.

## Hardware requirements
- 1x GPU with >=24GB VRAM (pod had 1x NVIDIA H100 80GB HBM3; `gemma-2-9b-it`
  in fp16 needs ~18GB for weights plus KV cache headroom).
- CUDA-capable PyTorch (pod had `torch==2.11.0+cu128`).

## Install
```bash
pip install -r requirements.txt
```

## Intended run commands
```bash
export HF_TOKEN=<token with accepted google/gemma-2-9b-it license>
export HF_HOME=/workspace/datasets/hf

# GSM8K (generative, 8-shot CoT via lm-evaluation-harness default config)
python eval.py --task gsm8k --model google/gemma-2-9b-it --dtype float16 \
    --output_dir /workspace/output/gsm8k_generations \
    --log_dir /workspace/output/eval_logs

# MMLU (5-shot, loglikelihood over answer choices)
python eval.py --task mmlu --model google/gemma-2-9b-it --dtype float16 \
    --output_dir /workspace/output/eval_logs/mmlu \
    --log_dir /workspace/output/eval_logs
```

Random seed: none pinned by `TASK.yaml.subject` (`seeds: []`, `n_seeds: 1`) —
`eval.py` defaults to `--seed 0` for determinism of few-shot example sampling.

Datasets: `openai/gsm8k` (config `main`) and `cais/mmlu` (config `all`) via
the Hugging Face `datasets` library, cached under `/workspace/datasets/hf`
(already downloaded successfully during this run — these are not gated).

`average_kv_bits` for this baseline is fixed at `16` by construction (FP16
storage, no quantization applied) and does not require executing the model
to determine.
