#!/usr/bin/env bash
# setup.sh  —  One-time setup for audio-tools.
# Run once after cloning:  bash setup.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ARCH="$(uname -m)"   # arm64 | x86_64
OS="$OSTYPE"         # darwin* | linux-gnu

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║          audio-tools  setup              ║"
echo "╚══════════════════════════════════════════╝"
echo "  Platform: $OS  arch: $ARCH"

# ── 1. System dependencies ──────────────────────────────────
echo ""
echo "[ 1 / 3 ]  System dependencies"

if [[ "$OS" == "darwin"* ]]; then
    if ! command -v brew &>/dev/null; then
        echo "  ✗  Homebrew not found."
        echo "     Install it from https://brew.sh then re-run this script."
        exit 1
    fi
    echo "  Installing python@3.11, rubberband, ffmpeg via Homebrew …"
    brew install python@3.11 rubberband ffmpeg

    # On Apple Silicon, Homebrew lives in /opt/homebrew and ships arm64 wheels.
    # On Intel, it lives in /usr/local — its Python targets macOS 10.15 which
    # is too old for some pre-built wheels (e.g. llvmlite needs macosx_12_0).
    # Setting MACOSX_DEPLOYMENT_TARGET=12.0 makes pip find the correct wheels.
    if [[ "$ARCH" == "arm64" ]] && [[ -x "/opt/homebrew/bin/python3.11" ]]; then
        PYTHON311="/opt/homebrew/bin/python3.11"
    else
        PYTHON311="$(brew --prefix python@3.11)/bin/python3.11"
        export MACOSX_DEPLOYMENT_TARGET="12.0"
        echo "  Intel Mac detected — setting MACOSX_DEPLOYMENT_TARGET=12.0 for wheel compatibility"
    fi

else
    echo "  Linux ($ARCH) detected. Checking for required system packages …"
    MISSING=()
    command -v python3.11 &>/dev/null || MISSING+=("python3.11")
    command -v ffmpeg     &>/dev/null || MISSING+=("ffmpeg")
    # rubberband is 'rubberband-cli' on Debian/Ubuntu, 'rubberband' on others.
    command -v rubberband &>/dev/null || MISSING+=("rubberband-cli")
    if [ ${#MISSING[@]} -gt 0 ]; then
        echo "  Missing: ${MISSING[*]}"
        echo "  Install with:  sudo apt install ${MISSING[*]}"
        exit 1
    fi
    echo "  All system dependencies found."
    PYTHON311="python3.11"
fi

# ── 2. System Python — beat stabilizer deps ─────────────────
echo ""
echo "[ 2 / 3 ]  Beat stabilizer dependencies (system Python)"
python3 -m pip install -r requirements.txt --break-system-packages 2>/dev/null \
    || python3 -m pip install -r requirements.txt
echo "  ✓  Done"

# ── 3. Create demucs venv (Python 3.11) ─────────────────────
echo ""
echo "[ 3 / 3 ]  Creating demucs + Beat This! virtual environment (Python 3.11) …"
"$PYTHON311" -m venv venv_demucs
./venv_demucs/bin/pip install --upgrade pip --quiet
./venv_demucs/bin/pip install -r requirements_demucs.txt
# Beat This! (MIT code + weights) for beat/downbeat detection; runs on the same torch.
./venv_demucs/bin/pip install soundfile "beat_this @ git+https://github.com/CPJKU/beat_this.git"
echo "  ✓  venv_demucs ready"

# ── Done ────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════╗"
echo "║  ✓  Setup complete!                      ║"
echo "║                                          ║"
echo "║  CLI pipeline:                           ║"
echo "║    python3 pipeline.py -i song.wav       ║"
echo "║                                          ║"
echo "║  Individual tools:                       ║"
echo "║    python3 beat_stabilizer.py -i song.wav -o out.wav"
echo "║    ./venv_demucs/bin/python stem_splitter.py -i out.wav"
echo "╚══════════════════════════════════════════╝"
echo ""
