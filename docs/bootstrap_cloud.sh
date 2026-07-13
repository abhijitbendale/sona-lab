#!/bin/bash
# bootstrap_cloud.sh
# One-shot setup for a fresh Ubuntu 22.04 cloud GPU instance.
#
# Usage:
#   chmod +x bootstrap_cloud.sh
#   ./bootstrap_cloud.sh
#
# NOTE: This script will prompt you to reboot after driver install.
# After reboot, re-run with:
#   ./bootstrap_cloud.sh --post-reboot

set -e

# ─── Configuration ─────────────────────────────────────────────────────────────
CUDA_VERSION="12.8"
CUDA_APT_VERSION="12-8"
CONDA_PREFIX="$HOME/miniconda3"
WORK_DIR="$HOME/work"
REPO_URL="https://github.com/inworld-ai/tts.git"
REPO_DIR="$WORK_DIR/tts"
CONDA_ENV="inworld_tts"
PYTHON_VERSION="3.10"
GIT_NAME="${GIT_NAME:-}"       # set via env or prompted below
GIT_EMAIL="${GIT_EMAIL:-}"     # set via env or prompted below
# ───────────────────────────────────────────────────────────────────────────────

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; BLUE='\033[0;34m'; NC='\033[0m'
info()    { echo -e "${BLUE}[INFO]  $*${NC}"; }
success() { echo -e "${GREEN}[OK]    $*${NC}"; }
warn()    { echo -e "${YELLOW}[WARN]  $*${NC}"; }
die()     { echo -e "${RED}[ERROR] $*${NC}"; exit 1; }

POST_REBOOT="${1:-}"

# ─── 0) Post-reboot fast path ──────────────────────────────────────────────────
if [[ "$POST_REBOOT" == "--post-reboot" ]]; then
    info "Post-reboot mode: verifying NVIDIA driver and continuing from step 4."
    if ! command -v nvidia-smi &>/dev/null; then
        die "nvidia-smi not found. Driver install may have failed. Check /var/log/dpkg.log."
    fi
    nvidia-smi
    success "NVIDIA driver verified."
    # Jump directly to CUDA setup
    SKIP_TO_CUDA=1
fi

# ─── 1) Base OS and utilities ──────────────────────────────────────────────────
if [[ -z "$SKIP_TO_CUDA" ]]; then
    info "Updating packages and installing base utilities..."
    sudo apt update && sudo apt upgrade -y
    sudo apt install -y \
        build-essential \
        git curl wget unzip zip \
        ca-certificates gnupg lsb-release software-properties-common \
        pkg-config cmake ninja-build \
        htop tmux tree jq ffmpeg nvtop
    success "Base utilities installed."
fi

# ─── 2) Git config ────────────────────────────────────────────────────────────
info "Configuring git..."
if [[ -z "$GIT_NAME" ]]; then
    read -r -p "  Git user.name: " GIT_NAME
fi
if [[ -z "$GIT_EMAIL" ]]; then
    read -r -p "  Git user.email: " GIT_EMAIL
fi
git config --global user.name  "$GIT_NAME"
git config --global user.email "$GIT_EMAIL"
success "Git configured as: $GIT_NAME <$GIT_EMAIL>"

# ─── 2b) SSH key for GitHub ────────────────────────────────────────────────────
SSH_KEY="$HOME/.ssh/id_ed25519"
if [[ ! -f "$SSH_KEY" ]]; then
    info "Generating SSH key..."
    ssh-keygen -t ed25519 -C "$GIT_EMAIL" -f "$SSH_KEY" -N ""
    success "SSH key generated at $SSH_KEY"
    echo ""
    echo "  ╔══════════════════════════════════════════════════════════════════╗"
    echo "  ║  Add this public key to your GitHub account before continuing:  ║"
    echo "  ║  https://github.com/settings/keys                               ║"
    echo "  ╚══════════════════════════════════════════════════════════════════╝"
    cat "$SSH_KEY.pub"
    echo ""
    read -r -p "  Press ENTER after adding the key to GitHub..."
else
    info "SSH key already exists at $SSH_KEY, skipping."
fi

ssh -T git@github.com 2>&1 | grep -qE 'successfully|verified' && success "GitHub SSH auth OK." \
    || warn "GitHub SSH test returned unexpected output. Continuing anyway."

# ─── 3) NVIDIA driver ─────────────────────────────────────────────────────────
if [[ -z "$SKIP_TO_CUDA" ]]; then
    if command -v nvidia-smi &>/dev/null; then
        info "NVIDIA driver already installed:"
        nvidia-smi
    else
        info "Installing NVIDIA driver (recommended)..."
        sudo apt install -y ubuntu-drivers-common
        RECOMMENDED=$(ubuntu-drivers devices 2>/dev/null | grep recommended | awk '{print $3}' | head -1)
        if [[ -n "$RECOMMENDED" ]]; then
            info "Installing recommended driver: $RECOMMENDED"
            sudo apt install -y "$RECOMMENDED"
        else
            warn "No recommended driver found via ubuntu-drivers. Running autoinstall..."
            sudo ubuntu-drivers autoinstall
        fi
        echo ""
        warn "Driver installed. A reboot is required before CUDA will work."
        warn "After reboot, re-run this script with: ./bootstrap_cloud.sh --post-reboot"
        read -r -p "  Reboot now? [y/N] " REBOOT_CONFIRM
        if [[ "$REBOOT_CONFIRM" =~ ^[Yy]$ ]]; then
            sudo reboot
        else
            warn "Skipping reboot. Continue with --post-reboot flag after manual reboot."
            exit 0
        fi
    fi
fi

# ─── 4) CUDA toolkit ──────────────────────────────────────────────────────────
if ! command -v nvcc &>/dev/null; then
    info "Installing CUDA $CUDA_VERSION toolkit..."
    wget -q -O /tmp/cuda-keyring.deb \
        https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i /tmp/cuda-keyring.deb
    sudo apt update
    sudo apt install -y "cuda-toolkit-${CUDA_APT_VERSION}"
    {
        echo "export PATH=/usr/local/cuda-${CUDA_VERSION}/bin:\$PATH"
        echo "export LD_LIBRARY_PATH=/usr/local/cuda-${CUDA_VERSION}/lib64:\$LD_LIBRARY_PATH"
    } >> ~/.bashrc
    # shellcheck disable=SC1090
    source ~/.bashrc
    success "CUDA $CUDA_VERSION installed."
else
    info "nvcc already found: $(nvcc --version | head -1)"
fi

# ─── 5) Miniconda ─────────────────────────────────────────────────────────────
if [[ ! -d "$CONDA_PREFIX" ]]; then
    info "Installing Miniconda..."
    wget -q -O /tmp/miniconda.sh \
        https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash /tmp/miniconda.sh -b -p "$CONDA_PREFIX"
    "$CONDA_PREFIX/bin/conda" init bash
    # shellcheck disable=SC1090
    source ~/.bashrc
    success "Miniconda installed at $CONDA_PREFIX."
else
    info "Miniconda already present at $CONDA_PREFIX."
fi

# Make conda available in this shell session
export PATH="$CONDA_PREFIX/bin:$PATH"
# shellcheck disable=SC1090
source "$CONDA_PREFIX/etc/profile.d/conda.sh"

# ─── 6) Clone repository ──────────────────────────────────────────────────────
mkdir -p "$WORK_DIR"
if [[ ! -d "$REPO_DIR/.git" ]]; then
    info "Cloning repository to $REPO_DIR..."
    git clone "$REPO_URL" "$REPO_DIR"
    success "Repository cloned."
else
    info "Repository already exists at $REPO_DIR. Pulling latest..."
    git -C "$REPO_DIR" pull --ff-only || warn "Could not fast-forward. Check git status."
fi

# ─── 7) Python environment ────────────────────────────────────────────────────
if ! conda env list | grep -q "^${CONDA_ENV} "; then
    info "Creating conda env '$CONDA_ENV' with Python $PYTHON_VERSION..."
    conda create -n "$CONDA_ENV" python="$PYTHON_VERSION" -y
    success "Conda env '$CONDA_ENV' created."
else
    info "Conda env '$CONDA_ENV' already exists."
fi

# ─── 8) Install project dependencies ─────────────────────────────────────────
info "Installing project dependencies into '$CONDA_ENV'..."
cd "$REPO_DIR"
conda run -n "$CONDA_ENV" bash setup/setup_python.sh
success "Project dependencies installed."

# ─── 9) HuggingFace login ─────────────────────────────────────────────────────
info "Hugging Face login (needed for gated models like Llama)."
warn "Do NOT enter your token here if you are worried about it being logged."
warn "You can skip and run 'hf auth login' manually in the conda env later."
read -r -p "  Log in to HuggingFace now? [y/N] " HF_LOGIN
if [[ "$HF_LOGIN" =~ ^[Yy]$ ]]; then
    conda run -n "$CONDA_ENV" python -c "from huggingface_hub import login; login()"
else
    warn "Skipped. Run 'conda activate $CONDA_ENV && hf auth login' before downloading gated models."
fi

# ─── 10) Quick validation ─────────────────────────────────────────────────────
info "Running quick validation..."
conda run -n "$CONDA_ENV" python - <<'PY'
import torch, torchaudio
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("torchaudio:", torchaudio.__version__)
PY

echo ""
success "Bootstrap complete!"
echo ""
echo "  Next steps:"
echo "    1. Activate env:     conda activate $CONDA_ENV"
echo "    2. Download models:  see docs/cpu_to_cloud_runbook.md section 5"
echo "    3. Dry run check:    python tts/training/main.py --config_path=example/configs/sft.json --dry_run"
echo "    4. Smoke train:      python tts/training/main.py --config_path=example/configs/sft_smoke.json --run_name=sft_smoke"
echo ""
echo "  To monitor GPU during training:"
echo "    watch -n 1 nvidia-smi"
echo ""
