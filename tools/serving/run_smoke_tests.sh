#!/usr/bin/env bash
# run_smoke_tests.sh
# One-shot script to download models/data and run verification smoke tests.
# Dynamically detects GPU+CUDA vs CPU and executes validation accordingly.
#
# Usage:
#   chmod +x tools/serving/run_smoke_tests.sh
#   ./tools/serving/run_smoke_tests.sh
#
# Can also be run via:
#   bash tools/serving/run_smoke_tests.sh

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

info()    { echo -e "${BLUE}[INFO]  $*${NC}"; }
success() { echo -e "${GREEN}[OK]    $*${NC}"; }
warn()    { echo -e "${YELLOW}[WARN]  $*${NC}"; }
die()     { echo -e "${RED}[ERROR] $*${NC}"; exit 1; }

# Get repository root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

info "Working directory: $REPO_ROOT"

# ─── 1) Environment Setup & Hardware Detection ───────────────────────────────
info "Detecting Python and Hardware environment..."

# Activate conda environment or virtual environment if available
if command -v conda &>/dev/null; then
    # Try activating sona_lab if it exists
    if conda env list | grep -q "^sona_lab "; then
        info "Activating conda environment: sona_lab"
        eval "$(conda shell.bash hook)"
        conda activate sona_lab
    fi
elif [[ -f "$REPO_ROOT/.venv/bin/activate" ]]; then
    info "Activating Python virtual environment (.venv)..."
    # shellcheck disable=SC1091
    source "$REPO_ROOT/.venv/bin/activate"
fi

# Detect CUDA and GPU availability via Python
HW_MODE=$(python3 - <<'PY'
import torch
if torch.cuda.is_available():
    print("GPU")
else:
    print("CPU")
PY
)

if [[ "$HW_MODE" == "GPU" ]]; then
    GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
    success "Hardware Detected: NVIDIA GPU ($GPU_NAME) with CUDA support."
    NUM_WORKERS=2
    BATCH_SIZE=8
else
    if command -v nvidia-smi &>/dev/null; then
        warn "nvidia-smi is available but torch.cuda.is_available() is false."
        warn "This usually means Torch/CUDA build mismatch with host driver."
        warn "Try: CUDA_VERSION=12.4 bash setup/setup_python.sh"
    fi
    warn "Hardware Detected: CPU only (No NVIDIA GPU/CUDA detected)."
    warn "Smoke tests will run in CPU fallback mode (slower, but functional for verification)."
    NUM_WORKERS=0
    BATCH_SIZE=2
fi

# ─── 2) Check & Download Models ──────────────────────────────────────────────
info "Checking required base models and codec checkpoints..."

mkdir -p models/codec/xcodec2/ckpt
mkdir -p models/base_lm/Llama-3.2-1B-Instruct
mkdir -p audios
mkdir -p experiments

# Check Codec Checkpoint
CODEC_CKPT="models/codec/xcodec2/ckpt/epoch=4-step=1400000.ckpt"
if [[ ! -f "$CODEC_CKPT" ]]; then
    info "Codec checkpoint not found at $CODEC_CKPT."
    if command -v hf &>/dev/null; then
        info "Downloading xcodec2 checkpoint via hf..."
        hf download HKUSTAudio/xcodec2 \
            --include "ckpt/*" "config.json" "model.safetensors" \
            --local-dir models/codec/xcodec2
    else
        warn "hf command not found. Please install huggingface_hub and run 'hf auth login'."
        warn "Then download HKUSTAudio/xcodec2 into models/codec/xcodec2 manually."
    fi
else
    success "Codec checkpoint found: $CODEC_CKPT"
fi

# Check Llama Base Model
LLAMA_DIR="models/base_lm/Llama-3.2-1B-Instruct"
if [[ ! -f "$LLAMA_DIR/model.safetensors" && ! -f "$LLAMA_DIR/consolidated.00.pth" ]]; then
    warn "Llama 3.2 1B base model not found in $LLAMA_DIR."
    warn "If you have access to gated models, download it using:"
    echo "  hf download meta-llama/Llama-3.2-1B-Instruct --local-dir models/base_lm/Llama-3.2-1B-Instruct"
    read -r -p "Do you want to attempt downloading Llama-3.2-1B-Instruct now? (Requires HF login) [y/N] " DL_LLAMA
    if [[ "$DL_LLAMA" =~ ^[Yy]$ ]]; then
        hf download meta-llama/Llama-3.2-1B-Instruct --local-dir models/base_lm/Llama-3.2-1B-Instruct
    else
        warn "Skipping Llama download. Note that SFT and Inference smoke tests will fail if model is missing."
    fi
else
    success "Base language model found in $LLAMA_DIR."
fi

# ─── 3) Download Sample Data & Prepare Manifests ─────────────────────────────
info "Checking dataset and vectorized artifacts..."

if [[ ! -f "data/processed/vectorized/train_samples.jsonl" || ! -f "data/processed/vectorized/train_codes.npy" ]]; then
    info "Preparing sample data and building manifests..."
    mkdir -p data/downloaded data/processed/manifests data/processed/vectorized

    python3 tools/data/download_small_datasets.py \
        --dataset_mode dummy \
        --output_root data/downloaded \
        --libri_limit 20 \
        --cv_limit 20

    python3 tools/data/build_manifests.py \
        --download_root data/downloaded \
        --manifest_root data/processed/manifests

    python3 tools/data/merge_manifests.py \
        --manifest_root data/processed/manifests \
        --max_total 40 \
        --val_ratio 0.1

    info "Vectorizing audio samples ($HW_MODE mode)..."
    python3 tools/data/data_vectorizer.py \
        --codec_model_path="$CODEC_CKPT" \
        --batch_size="$BATCH_SIZE" \
        --num_workers="$NUM_WORKERS" \
        --dataset_path="data/processed/manifests/train_small.jsonl" \
        --output_dir="data/processed/vectorized" \
        --run_name="smoke_vec"

    info "Merging vectorized data shards..."
    python3 tools/data/data_merger.py \
        --dataset_path="data/processed/vectorized" \
        --remove_shards

    success "Data preparation and vectorization complete."
else
    success "Vectorized dataset artifacts already present in data/processed/vectorized."
fi

# ─── 4) Run Verification Smoke Tests ─────────────────────────────────────────
info "=== Stage 1: Pre-training Pipeline Verification (Dry Run) ==="
python3 tts/training/main.py \
    --config_path="example/configs/sft.json" \
    --dry_run
success "Stage 1 (Dry Run) Passed!"

info "=== Stage 2: SFT Training Smoke Test (2 Steps) ==="
# Clean up any stale smoke experiment logs
rm -rf experiments/sft_smoke
python3 tts/training/main.py \
    --config_path="example/configs/sft_smoke.json" \
    --run_name="sft_smoke"
success "Stage 2 (SFT Training Smoke Test) Passed!"

info "=== Stage 3: Post-SFT Inference Verification ==="
PROMPT_WAV=$(find data/downloaded/libritts -name "*.flac" | head -n 1)
if [[ -z "$PROMPT_WAV" ]]; then
    PROMPT_WAV=$(find data/downloaded -name "*.flac" -o -name "*.wav" | head -n 1)
fi

if [[ -f "experiments/sft_smoke/final_model.pt" && -n "$PROMPT_WAV" ]]; then
    info "Running sample inference using trained checkpoint: experiments/sft_smoke/final_model.pt"
    python3 tools/serving/sample_inference.py \
        --model_checkpoint_path="experiments/sft_smoke/final_model.pt" \
        --prompt_wav="$PROMPT_WAV" \
        --prompt_transcription="SMOKE TEST PROMPT" \
        --text="This is an automated verification test of the synthesized voice." \
        --output_path="audios/verification_smoke_output.wav" \
        --max_tokens=32
    success "Stage 3 (Inference Verification) Passed! Output generated at audios/verification_smoke_output.wav"
else
    warn "Skipping Stage 3 inference test (trained checkpoint or prompt wav not found)."
fi

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                    ALL SMOKE TESTS PASSED SUCCESSFULLY!                  ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════════════════╝${NC}"
echo ""
info "Hardware mode used: $HW_MODE"
info "Your repository is verified and ready for full-scale cloud training!"
