# Bare Cloud Setup Order (Ubuntu 22.04 + Conda)

This is the shortest reliable order to bring up a fresh cloud GPU VM for Sona Lab.

## 1) Base OS tools

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y \
  build-essential git curl wget unzip zip \
  ca-certificates gnupg lsb-release software-properties-common \
  pkg-config cmake ninja-build htop tmux tree jq ffmpeg
```

## 2) NVIDIA driver

```bash
sudo ubuntu-drivers devices
sudo ubuntu-drivers autoinstall
sudo reboot
```

After reboot:

```bash
nvidia-smi
```

## 3) CUDA 12.8 toolkit

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-8
echo 'export PATH=/usr/local/cuda-12.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
nvcc --version
```

## 4) Install Miniconda

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
$HOME/miniconda3/bin/conda init bash
source ~/.bashrc
```

## 5) Clone repository

```bash
mkdir -p ~/work && cd ~/work
git clone https://github.com/abhijitbendale/sona-lab.git tts
cd tts
```

## 6) Create and activate conda environment

```bash
conda create -n sona_lab python=3.10 -y
conda activate sona_lab
python --version
```

## 7) Install project dependencies

```bash
bash setup/setup_python.sh
```

## 8) Hugging Face login (required for gated model downloads)

```bash
pip install -U huggingface_hub
hf auth login
```

## 9) One-shot smoke pipeline (recommended first run)

```bash
chmod +x tools/serving/run_smoke_tests.sh
./tools/serving/run_smoke_tests.sh
```

## 10) Manual staged checks (alternative to smoke script)

```bash
python tts/training/main.py --config_path=example/configs/sft.json --dry_run
python tts/training/main.py --config_path=example/configs/sft_smoke.json --run_name=sft_smoke
python tools/serving/sample_inference.py --model_checkpoint_path=experiments/sft_smoke/final_model.pt --max_tokens 64
```

## 11) Full cloud training in tmux

```bash
tmux new -s tts
conda activate sona_lab
python tts/training/main.py --config_path=example/configs/sft.json --run_name=sft_cloud
```

Useful monitor command:

```bash
watch -n 1 nvidia-smi
```

## Optional: bootstrap script

You can use `docs/bootstrap_cloud.sh` for a one-shot bootstrap. If you use it,
set `REPO_URL` inside the script to your fork before running.
