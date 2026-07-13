# CPU to Cloud Runbook (Inworld TTS)

This runbook captures a clean local setup, the minimum stage checks before cloud, and the exact command sequence we validated.

## 1) What To Verify Before Cloud

Run these in order:

1. Data pipeline smoke test
- Confirm merged vectorized files exist in `data/processed/vectorized`:
  - `train_codes.npy`, `train_codes_index.npy`, `train_samples.jsonl`
  - `val_codes.npy`, `val_codes_index.npy`, `val_samples.jsonl`

2. SFT dry run (already validated)
- Command:
  - `python tts/training/main.py --config_path=example/configs/sft.json --dry_run`
- Success signal: `Dry run completed successfully!`

3. Tiny SFT smoke train (2 steps)
- Command:
  - `python tts/training/main.py --config_path=example/configs/sft_smoke.json --run_name=sft_smoke`
- Success signals:
  - `Maximum number of steps reached. Stopping the training.`
  - `experiments/sft_smoke/final_model.pt` exists

4. Post-SFT inference
- Command:
  - `python tools/serving/sample_inference.py --model_checkpoint_path=experiments/sft_smoke/final_model.pt --max_tokens 64`
- Expectation:
  - Script runs end-to-end and writes output wav (quality will be limited with tiny training).

5. RLHF smoke (optional on laptop, recommended on cloud)
- First validate config/paths on CPU with very small settings.
- Full RLHF run should be done on cloud GPU.

## 2) Clean Local Install (CPU)

Use this when you want a fresh local re-setup.

1. Create env
- `conda create -n inworld_tts python=3.10 -y`
- `conda activate inworld_tts`

2. Install CPU PyTorch stack
- `pip install torch==2.13.0+cpu torchaudio==2.11.0+cpu --index-url https://download.pytorch.org/whl/cpu`

3. Install project
- `pip install -e .`

4. Install packages that were required during smoke tests
- `pip install datasets huggingface_hub vector-quantize-pytorch omegaconf cattrs regex unidecode ftfy soundfile peft==0.19.1 lingua-language-detector==2.0.2 nemo_text_processing`

5. Optional: snapshot this working env
- `pip freeze > setup/requirements.cpu.lock.txt`

## 3) Known CPU Warnings (Safe To Ignore)

- `torchao` shared library warnings on CPU.
- `pin_memory` warning when no accelerator is present.
- `torchaudio/torchcodec` load warnings when fallback to `soundfile` succeeds.

## 4) Commands We Standardized

1. SFT dry run:
- `python tts/training/main.py --config_path=example/configs/sft.json --dry_run`

2. Tiny SFT train:
- `python tts/training/main.py --config_path=example/configs/sft_smoke.json --run_name=sft_smoke`

3. Post-SFT inference:
- `python tools/serving/sample_inference.py --model_checkpoint_path=experiments/sft_smoke/final_model.pt --max_tokens 64`

## 5) Cloud Readiness Checklist

Before spinning up cloud:

1. Confirm local artifacts
- `models/base_lm/Llama-3.2-1B-Instruct`
- `models/codec/xcodec2/ckpt/epoch=4-step=1400000.ckpt`
- `data/processed/vectorized/*`

2. Keep configs ready
- `example/configs/sft.json`
- `example/configs/sft_smoke.json`
- RLHF config you plan to use

3. Keep one known-good command per stage
- SFT train
- checkpoint conversion
- inference command
- RLHF launch command

4. Save environment lock
- `pip freeze > setup/requirements.cpu.lock.txt`
- On cloud, create a GPU-specific lock once stable.

## 6) Practical Notes

- CPU laptop is great for plumbing checks and tiny smoke runs.
- Real SFT quality and RLHF should be treated as cloud GPU tasks.
- If inference from base model fails before training, that is expected; use trained checkpoints for meaningful outputs.
