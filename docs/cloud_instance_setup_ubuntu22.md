# Cloud Instance Setup (Ubuntu 22.04)

This guide assumes a fresh Ubuntu 22.04 VM with a supported NVIDIA GPU and no ML stack preinstalled.

## 0) Conventions

- Run commands as a sudo-capable user.
- Reboot when instructed.
- Replace placeholders like `<your_user>` and `<repo_url>`.

## 1) Base OS and Utilities

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y \
  build-essential \
  git curl wget unzip zip \
  ca-certificates gnupg lsb-release software-properties-common \
  pkg-config cmake ninja-build \
  htop tmux tree jq ffmpeg
```

Optional but useful:

```bash
sudo apt install -y nvtop
```

## 2) Git and GitHub Access

Configure git:

```bash
git config --global user.name "<your_name>"
git config --global user.email "<your_email>"
```

Create SSH key (if needed):

```bash
ssh-keygen -t ed25519 -C "<your_email>"
cat ~/.ssh/id_ed25519.pub
```

Add that public key in GitHub settings, then test:

```bash
ssh -T git@github.com
```

## 3) Install NVIDIA Driver

Check GPU presence:

```bash
lspci | grep -Ei 'nvidia|vga|3d'
```

Install recommended driver:

```bash
sudo ubuntu-drivers devices
sudo ubuntu-drivers autoinstall
sudo reboot
```

After reboot, verify:

```bash
nvidia-smi
```

If `nvidia-smi` fails, fix driver first before proceeding.

## 4) Install CUDA Toolkit (12.8 recommended for this repo)

Install CUDA keyring and toolkit:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-8
```

Add CUDA to shell env:

```bash
echo 'export PATH=/usr/local/cuda-12.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

Verify:

```bash
nvcc --version
nvidia-smi
```

## 5) Install Miniconda

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
$HOME/miniconda3/bin/conda init bash
source ~/.bashrc
```

## 6) Clone Repository

```bash
mkdir -p ~/work && cd ~/work
git clone <repo_url> tts
cd tts
```

If using this upstream repo directly:

```bash
git clone https://github.com/inworld-ai/tts.git
cd tts
```

## 7) Create Python Environment

```bash
conda create -n inworld_tts python=3.10 -y
conda activate inworld_tts
python --version
```

## 8) Install Project Dependencies

### Option A (recommended): repo setup script using conda

```bash
conda activate inworld_tts
bash setup/setup_python.sh
```

### Option B: direct pip install

```bash
conda activate inworld_tts
pip install -U pip setuptools wheel
pip install -e .[cu128]
```

If flash-attn wheel is needed and not installed automatically, use the repo script path from `setup/setup_python.sh`.

## 9) Hugging Face Auth and Gated Models

Install HF CLI and login:

```bash
pip install -U huggingface_hub
hf auth login
```

Validate gated model access if needed:

```bash
python - <<'PY'
from huggingface_hub import model_info
print(model_info('meta-llama/Llama-3.2-1B-Instruct').id)
PY
```

## 10) Runtime Utilities for Long Jobs

Use tmux so jobs survive disconnect:

```bash
tmux new -s tts
```

Inside tmux, run training commands.

GPU monitoring:

```bash
watch -n 1 nvidia-smi
```

## 11) Quick Validation Checks

```bash
python - <<'PY'
import torch, torchaudio
print('torch', torch.__version__)
print('cuda available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu', torch.cuda.get_device_name(0))
print('torchaudio', torchaudio.__version__)
PY
```

From repo root, run a smoke config/dry run as needed:

```bash
python tts/training/main.py --config_path=example/configs/sft.json --dry_run
```

## 12) Suggested Folder Layout on Cloud

```text
~/work/tts                       # repo
~/work/tts/models                # model checkpoints
~/work/tts/data                  # datasets/vectorized artifacts
~/work/tts/experiments           # run outputs/checkpoints
~/work/tts/logs                  # optional redirected logs
```

## 13) Common Failure Notes

- Driver mismatch: reinstall or upgrade NVIDIA driver, then reboot.
- `torch.cuda.is_available() == False`: driver or CUDA runtime problem.
- Flash-attn install failures: align CUDA/PyTorch versions and use prebuilt wheel path from repo setup script.
- Gated model errors: ensure `hf auth login` token has access.

## 14) One-Time Bootstrap Script (Optional)

If you want, you can convert this guide into a single `bootstrap_cloud.sh` script, but keeping manual steps for driver/CUDA and reboot is safer on fresh VMs.
