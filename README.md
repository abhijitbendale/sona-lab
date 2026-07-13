# Sona Lab 🎙️⚡

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CUDA Support](https://img.shields.io/badge/CUDA-12.4%20%7C%2012.8-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6%20%7C%202.7-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Sona Lab** is an open neural speech synthesis, pre-training, fine-tuning, and alignment laboratory. Designed for experimental research and cloud training (e.g., Vast.ai), Sona Lab provides a robust, hardware-resilient pipeline for training SpeechLM models and 1D audio codecs across CPU laptops and multi-GPU cloud clusters.

---

## 🌟 Key Features

- **Hardware Resilience:** Dynamically adapts across CPU laptops (for plumbing checks and dry runs) and multi-GPU cloud clusters without code changes or crashing on missing C++ extensions.
- **End-to-End Data Pipeline:** Ready-to-use scripts to download sample data (LibriTTS/Common Voice), build manifests, vectorize audio into codec embeddings, and merge dataset shards.
- **Unified Training Loop:** Support for Supervised Fine-Tuning (SFT), Reinforcement Learning from Human Feedback (RLHF), and Pre-training using DDP, DeepSpeed, or FSDP.
- **Automated Verification Suite:** Built-in smoke tests (`run_smoke_tests.sh`) to validate dry runs, short SFT training, and inference before scaling up to cloud GPUs.
- **Cloud Readiness:** Includes automated bootstrap scripts (`bootstrap_cloud.sh`) and comprehensive runbooks for one-click deployment on fresh Ubuntu 22.04 cloud VMs.

---

## 📚 Documentation & Runbooks

We have documented the entire workflow from laptop experimentation to cloud scale in the `docs/` folder:

- **[CPU to Cloud Runbook](docs/cpu_to_cloud_runbook.md):** The standard reference guide for pre-cloud verification checks, local CPU env recreation, and standard commands.
- **[Bare Cloud Setup Order](docs/bare_cloud_setup.md):** Fast, ordered checklist for a fresh Ubuntu 22.04 GPU VM, including conda env creation and first smoke run.
- **[Cloud Instance Setup Guide](docs/cloud_instance_setup_ubuntu22.md):** Detailed step-by-step manual setup instructions for a fresh Ubuntu 22.04 VM (NVIDIA drivers, CUDA 12.8, Miniconda, HF login, and tmux).
- **[Cloud Bootstrap Script](docs/bootstrap_cloud.sh):** Automated one-shot shell script that configures drivers, CUDA toolkit, conda environments, SSH keys, and dependencies automatically.

---

## 🚀 Quick Start

### 1. One-Shot Automated Verification (Local or Cloud)

You can run our automated verification suite to check hardware, download small datasets and base checkpoints if missing, and run dry-run / SFT / inference smoke tests:

```bash
chmod +x tools/serving/run_smoke_tests.sh
./tools/serving/run_smoke_tests.sh
```

### 2. Manual Environment Setup

If you prefer manual installation:

```bash
# Create and activate environment
conda create -n sona_lab python=3.10 -y
conda activate sona_lab

# Install dependencies (Linux CUDA 12.8 recommended)
bash setup/setup_python.sh
source .venv/bin/activate
```

---

## 🎯 Training Workflow Overview

1. **Data Vectorization:** Convert raw `.wav`/`.flac` files and transcripts into discrete audio tokens using our codec encoder:
   ```bash
   torchrun --nproc_per_node 1 tools/data/data_vectorizer.py \
       --codec_model_path=models/codec/xcodec2/ckpt/epoch=4-step=1400000.ckpt \
       --batch_size=16 \
       --dataset_path=data/processed/manifests/train_small.jsonl \
       --output_dir=data/processed/vectorized --run_name=my_vec

   python tools/data/data_merger.py --dataset_path data/processed/vectorized --remove_shards
   ```
2. **Supervised Fine-Tuning (SFT):**
   ```bash
   python tts/training/main.py --config_path=example/configs/sft.json --run_name=my_sft_run
   ```
3. **Inference / Serving:**
   ```bash
   python tools/serving/sample_inference.py \
       --model_checkpoint_path=experiments/my_sft_run/final_model.pt \
       --prompt_wav=data/downloaded/libritts/libri_00014.flac \
       --text="Hello from Sona Lab!" \
       --output_path=audios/sona_output.wav
   ```

---

## 🤝 Acknowledgements & Archival Notice

**Sona Lab** originally began as an experimental fork and architectural extension of the open-source **[Inworld TTS](https://github.com/inworld-ai/tts)** repository created by the Inworld AI team.

We extend our gratitude to Inworld AI, the LLaMA / Meta AI team, and the PyTorch / Hugging Face open-source communities. For archival and historical reference, the original upstream Inworld README has been preserved in this repository as [`README_INWORLD_ARCHIVE.md`](README_INWORLD_ARCHIVE.md).

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.
