**BLOCKED**: this experiment requires `google/gemma-2-9b-it`, a gated Hugging Face model. No HF token with an accepted license was available in this run's environment, so this code has NOT been executed end-to-end against the pinned model. `quantization.py`'s self-test and `smoke_test_cache_patch.py` (on an ungated proxy model) WERE run and passed — see their output below.

## What this is

Baseline "uniform 2-bit KV-cache quantization" code for the attention-stability-boundary / certified KV-cache quantization research line. It fake-quantizes every key/value element written into (and read back out of) `google/gemma-2-9b-it`'s `DynamicCache` to 2 bits (asymmetric min-max, per-token grouping), and evaluates the resulting model on GSM8K and MMLU. This is the uniform-precision baseline that the project's adaptive/certified-boundary method (ASB-KVQ) is compared against.

## Files

- `quantization.py` — standalone, architecture-agnostic N-bit fake-quantization (`quantize_dequantize`). Has a `__main__` self-test.
- `cache_patch.py` — class-level monkeypatch of `transformers.cache_utils.DynamicLayer.update` / `DynamicSlidingWindowLayer.update` that quantizes K/V states before they're cached. `install_uniform_kv_quant(model, num_bits=2)` / `uninstall_uniform_kv_quant()` / `uniform_kv_quant(model, num_bits=2)` context manager.
- `smoke_test_cache_patch.py` — smoke test of the patch mechanism on an ungated proxy model (does NOT touch gemma-2-9b-it).
- `eval_gsm8k.py` — GSM8K harness (4-shot chain-of-thought, greedy decoding, exact-match on `#### <number>`).
- `eval_mmlu.py` — MMLU harness (5-shot in-subject exemplars, next-token log-prob over A/B/C/D).
- `run_baseline.py` — orchestrator: loads the model once, installs the patch, runs both evals, writes `eval_logs/baseline_summary.json`.
- `requirements.txt` — pinned dependency versions.

## Hardware requirements

1x GPU with >= 24GB VRAM, bf16 support. Developed/intended for 1x NVIDIA H100 80GB (the platform's assigned GPU for this experiment).

## Install

```bash
pip install -r requirements.txt
```

## Authentication (required before running against the real model)

`google/gemma-2-9b-it` is gated. To run this code for real:

1. Accept the license at https://huggingface.co/google/gemma-2-9b-it
2. Authenticate: `huggingface-cli login`, or `export HF_TOKEN=<your_token>`

## Run command

```bash
python run_baseline.py
```

(Optional flags: `--n_samples N` to subsample for a quick smoke check, `--max_new_tokens`, `--num_bits`, `--model_name`.)

Individual harnesses can also be run standalone: `python eval_gsm8k.py`, `python eval_mmlu.py` (each loads the model itself when run directly; each also exposes `run_gsm8k(model, tokenizer, ...)` / `run_mmlu(model, tokenizer, ...)` for reuse without reloading, which is what `run_baseline.py` uses).

## Seed / determinism

`TASK.yaml` for this experiment declares `n_seeds=1` with an empty `seeds` list. Decoding is fully greedy (`do_sample=False`) throughout, and MMLU scoring is a single deterministic forward pass (no sampling at all) — so no random seed is needed or set; the pipeline is deterministic given fixed weights and inputs.

## Dataset provenance

- `openai/gsm8k` ("main" config, "test" split, 1319 examples) via Hugging Face `datasets`.
- `cais/mmlu` ("all" config, "test" split = 14042 rows, "dev" split = 285 rows used for 5-shot exemplars) via Hugging Face `datasets`.

Both are already cached under `$HF_HOME` (`/workspace/datasets/hf` on this platform) and load offline; the harnesses rely on the `HF_HOME` env var / default HF caching rather than hardcoding a `cache_dir`.

## What WAS verified in this run (no gated model needed)

### 1. `quantization.py` self-test — exact stdout

```
$ python quantization.py
[a] hand-checkable round-trip test passed:
    input:
 tensor([[[[ 0.,  1.,  2.,  3.],
          [-1.,  0.,  1.,  2.],
          [10., 10., 10., 10.]]]])
    dequantized:
 tensor([[[[ 0.,  1.,  2.,  3.],
          [-1.,  0.,  1.,  2.],
          [10., 10., 10., 10.]]]])
[b] mean abs error by bit-width: {2: 0.3782309591770172, 4: 0.07546819746494293, 8: 0.004434567876160145}
[PASS]
```

### 2. `smoke_test_cache_patch.py` — exact stdout

Proxy model used: `hf-internal-testing/tiny-random-Gemma2ForCausalLM` (first candidate tried; confirmed to use `DynamicCache` with `DynamicLayer`/`DynamicSlidingWindowLayer` instances before being accepted as a valid proxy).

```
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
[transformers] `torch_dtype` is deprecated! Use `dtype` instead!
[smoke_test] attempting to load candidate proxy model: hf-internal-testing/tiny-random-Gemma2ForCausalLM
Loading weights:   0%|          | 0/13 [00:00<?, ?it/s]Loading weights: 100%|██████████| 13/13 [00:00<00:00, 26728.41it/s]
[smoke_test]   type(past_key_values) = <class 'transformers.cache_utils.DynamicCache'>
[smoke_test]   all layers are DynamicLayer/DynamicSlidingWindowLayer: True
[smoke_test] SELECTED proxy model: hf-internal-testing/tiny-random-Gemma2ForCausalLM (confirmed DynamicCache/DynamicLayer usage)
[smoke_test] baseline (unpatched) generated ids: [2, 651, 4320, 8426, 25341, 83223, 83223, 83223, 254887, 254887, 254887, 254887, 254887]
[cache_patch] installing uniform 2-bit KV fake-quant patch (class-level, global) for model='hf-internal-testing/tiny-random-Gemma2ForCausalLM'
[smoke_test] patched (2-bit KV quant) generated ids:   [2, 651, 4320, 8426, 25341, 48290, 93673, 93673, 93673, 93673, 93673, 93673, 93673]
[cache_patch] uninstalled uniform 2-bit KV fake-quant patch, restored original DynamicLayer/DynamicSlidingWindowLayer.update
[smoke_test] restored (post-uninstall) generated ids:      [2, 651, 4320, 8426, 25341, 83223, 83223, 83223, 254887, 254887, 254887, 254887, 254887]
[smoke_test] proxy model used: hf-internal-testing/tiny-random-Gemma2ForCausalLM
[smoke_test] patch altered output relative to baseline: True
[smoke_test] uninstall cleanly restored baseline output: True
[PASS] cache_patch mechanism verified on proxy model hf-internal-testing/tiny-random-Gemma2ForCausalLM
```

This confirms: (i) the patched generation differs numerically from the unpatched baseline (the quantization has a real effect on attention/generation), and (ii) after `uninstall_uniform_kv_quant()`, generation exactly reproduces the original unpatched baseline (the patch is cleanly reversible and does not leak state).

### 3. Syntax / import checks

`python -m py_compile <file>` exits 0 for all of: `quantization.py`, `cache_patch.py`, `smoke_test_cache_patch.py`, `eval_gsm8k.py`, `eval_mmlu.py`, `run_baseline.py`. `import cache_patch` succeeds without a GPU or network access.

## What was NOT run

`eval_gsm8k.py`, `eval_mmlu.py`, and `run_baseline.py` were NOT executed against `google/gemma-2-9b-it` — this is confirmed to fail with an HTTP 401 gated-repo error in this environment (no HF_TOKEN with accepted license). No workaround was attempted, per the experiment's explicit operating instructions. Once a valid token is available, `python run_baseline.py` should run the full pipeline end-to-end as described above.
