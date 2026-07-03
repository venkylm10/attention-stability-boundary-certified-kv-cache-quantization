# baseline-methods — Gemma-2-9B-It KV-cache quantization baselines

Establishes FP16 and standard KV-cache-quantization baseline numbers
(gsm8k_accuracy, mmlu_accuracy, decoding_latency, kv_cache_size) on
`google/gemma-2-9b-it`, to anchor the sibling `main-asb-kvq` and ablation
experiments in this project.

## Status: BLOCKED — not executed

This run could not execute: `google/gemma-2-9b-it` is a **gated** model on
HuggingFace Hub and no `HF_TOKEN` (or equivalent) with the Gemma-2 license
accepted was available in the pod environment. See `/workspace/output/report.md`
for full details. The code below is complete and ready to run as soon as a
valid token is provisioned — nothing here was substituted for a different
model.

## Hardware requirements

- 1x GPU with >= 24GB VRAM (tested target: 1x NVIDIA H100 80GB)
- CUDA 12.8+, `torch==2.11.0+cu128` (pre-installed in the target pod image)
- No `flash-attn` required — falls back to PyTorch `sdpa` attention
  (`attn_implementation="sdpa"`), since no prebuilt flash-attn wheel/nvcc
  toolchain was available in this pod image.

## Install

```bash
pip install -r requirements.txt
```

## Authenticate (required — do this before running)

```bash
huggingface-cli login   # or: export HF_TOKEN=<your token>
# Accept the license at https://huggingface.co/google/gemma-2-9b-it first.
```

## Run

```bash
python train.py --n_gsm8k 200 --n_mmlu 500 --out_dir /workspace/output
```

This runs all 5 methods (`fp16`, `uniform2bit`, `gear`, `rotatekv`, `kitty`)
sequentially against the loaded model (loaded once, methods swapped via a
custom `past_key_values` cache — see `quant_methods.py`), writing:

- `baseline_logs/<method>_{gsm8k,mmlu}.log` — per-example running-accuracy logs
- `baseline_generations/<method>_{gsm8k,mmlu}.jsonl` — raw generations/predictions
- `baseline_raw_results.json` — per-method metrics, to be assembled into the
  enriched `results.json` manifest by the orchestrating agent.

Random seed: `42` (pinned per `TASK.yaml.subject.seeds`), `n_seeds=1`.

## Datasets

- `openai/gsm8k` (config `main`, test split, 1319 rows) — 8-shot CoT, greedy
  decoding, exact-match on the final numeric answer.
- `cais/mmlu` (config `all`, test split, 14042 rows) — 5-shot, scored via
  next-token log-likelihood over the four answer-letter tokens (standard
  Hendrycks et al. MMLU protocol).

Both datasets were successfully downloaded and cached during environment
setup for this run (`/workspace/datasets/hf`, ~218MB) — only the model is
blocked.

## Methods (see `quant_methods.py` docstring for full detail)

`fp16` (reference), `uniform2bit`, `gear` (arXiv:2403.05527 — quantize +
low-rank residual + sparse outlier correction), `rotatekv` (orthogonal
rotation before quantization to decorrelate outlier channels), `kitty`
(arXiv:2511.18643 — per-channel variance-ranked mixed 2-bit/4-bit precision).
All are faithful reimplementations of each paper's core algorithm using
fake-quantization (dequantize-in-place) rather than the papers' custom
CUDA/Triton kernels, since those kernels' reference implementations don't
build in this pod image (no CUDA toolkit/nvcc for source builds). Real KV
cache memory footprint is computed separately as a deterministic function of
architecture + bit-width in `kv_cache_size.py`, not measured via live GPU
allocator stats (which would conflate cache size with activation memory).
