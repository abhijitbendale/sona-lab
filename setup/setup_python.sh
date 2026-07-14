#!/bin/bash
# Setup script for Sona Lab project
# Uses standard pip / conda environment installation
#
# CUDA Version Support:
# Set CUDA_VERSION environment variable to specify CUDA version (12.4, 12.8)
# Default: 12.8
# Installs flash-attn prebuild wheels from https://github.com/mjun0812/flash-attention-prebuild-wheels
#
# Examples:
#   ./setup_python.sh                    # Uses CUDA 12.8 + PyTorch 2.7 + flash-attn 2.8.1
#   CUDA_VERSION=12.4 ./setup_python.sh  # Uses CUDA 12.4 + PyTorch 2.6 + flash-attn 2.8.0
#   CUDA_VERSION=12.8 ./setup_python.sh  # Uses CUDA 12.8 + PyTorch 2.7 + flash-attn 2.8.1

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Emojis for better UX
ROCKET="🚀"
PACKAGE="📦"
CHECK="✅"
CROSS="❌"
WARNING="⚠️"
TRASH="🗑️"
FOLDER="📁"
LIGHTBULB="💡"
PARTY="🎉"
CLIPBOARD="📋"

log_info() {
    echo -e "${BLUE}${PACKAGE} $1${NC}"
}

log_success() {
    echo -e "${GREEN}${CHECK} $1${NC}"
}

log_error() {
    echo -e "${RED}${CROSS} $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}${WARNING} $1${NC}"
}

log_header() {
    echo -e "${BLUE}${ROCKET} $1${NC}"
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

get_torch_build_info() {
    python3 - <<'PY'
try:
    import torch
    print(f"{torch.__version__}|{torch.version.cuda or 'None'}")
except Exception:
    print("MISSING|None")
PY
}

ensure_torch_cuda_compat() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        return 0
    fi

    CURRENT_INFO=$(get_torch_build_info)
    CURRENT_TORCH_VERSION="${CURRENT_INFO%%|*}"
    CURRENT_TORCH_CUDA="${CURRENT_INFO##*|}"

    TORCH_MISSING=0
    TORCH_MISMATCH=0
    if [[ "$CURRENT_TORCH_VERSION" == "MISSING" ]]; then
        TORCH_MISSING=1
    elif [[ "$CURRENT_TORCH_VERSION" != ${TORCH_SERIES}* || "$CURRENT_TORCH_CUDA" != "$TARGET_TORCH_CUDA" ]]; then
        TORCH_MISMATCH=1
    fi

    if [[ "$TORCH_MISSING" -eq 1 || "$TORCH_MISMATCH" -eq 1 ]]; then
        if [[ "$TORCH_MISSING" -eq 1 ]]; then
            log_warning "PyTorch not found. Installing pinned Torch stack for CUDA $CUDA_VERSION..."
        else
            log_warning "Detected incompatible Torch build: version=$CURRENT_TORCH_VERSION cuda=$CURRENT_TORCH_CUDA"
            log_warning "Target build is Torch $TORCH_PIN_VERSION with CUDA $TARGET_TORCH_CUDA. Reinstalling clean stack..."
        fi

        python3 -m pip uninstall -y torch torchaudio torchvision flash-attn >/dev/null 2>&1 || true
        python3 -m pip install --no-cache-dir --index-url "$TORCH_INDEX_URL" \
            "torch==$TORCH_PIN_VERSION" "torchaudio==$TORCHAUDIO_PIN_VERSION"

        UPDATED_INFO=$(get_torch_build_info)
        UPDATED_TORCH_VERSION="${UPDATED_INFO%%|*}"
        UPDATED_TORCH_CUDA="${UPDATED_INFO##*|}"
        log_success "Torch stack set to version=$UPDATED_TORCH_VERSION cuda=$UPDATED_TORCH_CUDA"
    else
        log_success "Torch/CUDA pre-check passed (version=$CURRENT_TORCH_VERSION, cuda=$CURRENT_TORCH_CUDA)"
    fi
}

# Get the project root directory (parent of setup directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

log_header "Starting Sona Lab Python setup..."
echo -e "${FOLDER} Working in: $PROJECT_ROOT"

# Show CUDA version that will be used
CUDA_VERSION="${CUDA_VERSION:-12.8}"
if [[ "$OSTYPE" == "darwin"* ]]; then
    log_info "Platform: macOS (will use CPU-only PyTorch)"
else
    log_info "Platform: Linux (will use CUDA $CUDA_VERSION)"
fi

# Ensure pip is installed and upgraded
log_info "Upgrading pip and setuptools..."
python3 -m pip install --upgrade pip setuptools wheel

# Change to project root
cd "$PROJECT_ROOT"

# Step 1: Install project dependencies
log_info "Installing project dependencies..."

# Install with CUDA extra on Linux, without extra on macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    log_info "Installing project dependencies (macOS - CPU only)..."
    INSTALL_CMD="python3 -m pip install -e ."
else
    # Map CUDA version to extra (cu124 and cu128)
    case "$CUDA_VERSION" in
        "12.4")
            CUDA_EXTRA="cu124"
            TORCH_VERSION="2.6"
            TORCH_SERIES="2.6"
            TORCH_PIN_VERSION="2.6.0"
            TORCHAUDIO_PIN_VERSION="2.6.0"
            TARGET_TORCH_CUDA="12.4"
            TORCH_INDEX_URL="https://download.pytorch.org/whl/cu124"
            ;;
        "12.8")
            CUDA_EXTRA="cu128"
            TORCH_VERSION="2.7"
            TORCH_SERIES="2.7"
            TORCH_PIN_VERSION="2.7.0"
            TORCHAUDIO_PIN_VERSION="2.7.0"
            TARGET_TORCH_CUDA="12.8"
            TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"
            ;;
        *)
            log_warning "Unsupported CUDA version: $CUDA_VERSION. Only CUDA 12.4 and 12.8 are supported. Defaulting to CUDA 12.8"
            CUDA_EXTRA="cu128"
            TORCH_VERSION="2.7"
            TORCH_SERIES="2.7"
            TORCH_PIN_VERSION="2.7.0"
            TORCHAUDIO_PIN_VERSION="2.7.0"
            TARGET_TORCH_CUDA="12.8"
            TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"
            ;;
    esac

    log_info "Installing project dependencies with CUDA extra: $CUDA_EXTRA..."
    INSTALL_CMD="python3 -m pip install -e .[$CUDA_EXTRA]"
fi

if $INSTALL_CMD; then
    log_success "Project dependencies installed successfully"
else
    log_error "Failed to install project dependencies"
    exit 1
fi

# Step 1b: Ensure Torch build matches selected CUDA version
if [[ "$OSTYPE" != "darwin"* ]]; then
    log_info "Running Torch/CUDA compatibility pre-check and auto-repair..."
    ensure_torch_cuda_compat
fi

# Step 2: Install flash-attn prebuild wheel (Linux only)
if [[ "$OSTYPE" != "darwin"* ]]; then
    log_info "Installing flash-attn prebuild wheel for CUDA $CUDA_VERSION + PyTorch $TORCH_VERSION..."

    # Determine Python version
    PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    PYTHON_VERSION_NODOT=$(echo "$PYTHON_VERSION" | tr -d '.')

    # Construct flash-attn wheel URL
    case "$CUDA_VERSION" in
        "12.4")
            PREBUILD_VERSION="0.3.12"
            FLASH_ATTN_VERSION="2.8.0"
            ;;
        "12.8")
            PREBUILD_VERSION="0.3.13"
            FLASH_ATTN_VERSION="2.8.1"
            ;;
    esac

    FLASH_ATTN_WHEEL="flash_attn-${FLASH_ATTN_VERSION}+cu${CUDA_VERSION//./}torch${TORCH_VERSION}-cp${PYTHON_VERSION_NODOT}-cp${PYTHON_VERSION_NODOT}-linux_x86_64.whl"
    FLASH_ATTN_URL="https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v${PREBUILD_VERSION}/${FLASH_ATTN_WHEEL}"

    log_info "Installing: $FLASH_ATTN_WHEEL"

    if python3 -m pip install "$FLASH_ATTN_URL"; then
        log_success "flash-attn prebuild wheel installed successfully"
    else
        log_warning "flash-attn prebuild wheel installation failed, but continuing setup..."
        echo -e "${LIGHTBULB} You can install it manually later with:"
        echo "   pip install $FLASH_ATTN_URL"
        echo -e "${LIGHTBULB} Prebuild wheels available at: https://github.com/mjun0812/flash-attention-prebuild-wheels"
    fi
else
    log_warning "Skipping NeMo-text-processing installation on macOS"
    log_warning "Skipping flash-attn prebuild wheel on macOS (use pip install flash-attn for CPU version if needed)"
fi

echo ""
log_success "Setup completed successfully!"
echo ""
echo -e "${CLIPBOARD} Next steps:"
echo "   1. Ensure your conda environment is active"
echo "   2. Start developing!"
echo ""
echo -e "${LIGHTBULB} CUDA version info:"
echo "   • Current CUDA version: $CUDA_VERSION"
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "   • PyTorch version: $TORCH_VERSION"
    echo "   • Installed with extra: $CUDA_EXTRA"
    echo "   • flash-attn: prebuild wheel from mjun0812/flash-attention-prebuild-wheels"
    echo "   • Supported CUDA versions:"
    echo "     CUDA_VERSION=12.4 $0  # CUDA 12.4 + PyTorch 2.6 + flash-attn 2.8.0"
    echo "     CUDA_VERSION=12.8 $0  # CUDA 12.8 + PyTorch 2.7 + flash-attn 2.8.1"
else
    echo "   • macOS detected - using CPU-only PyTorch"
    echo "   • For flash-attn on macOS: pip install flash-attn (CPU version)"
fi
echo ""
echo -e "${PARTY} Happy coding!"
