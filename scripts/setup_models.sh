#!/usr/bin/env bash
set -euo pipefail

SADTALKER_DIR="${SADTALKER_ROOT:-/content/SadTalker}"

echo "Installing system packages (ffmpeg, espeak-ng, git)..."
if command -v apt-get >/dev/null 2>&1; then
    apt-get update -qq
    apt-get install -y -qq ffmpeg espeak-ng espeak git libportaudio2
fi

echo "Cloning SadTalker and downloading checkpoints..."
if [ ! -d "$SADTALKER_DIR" ]; then
    git clone -q https://github.com/OpenTalker/SadTalker.git "$SADTALKER_DIR"
fi
(cd "$SADTALKER_DIR" && bash scripts/download_models.sh)

echo "Downloading TinyLlama-1.1B-Chat-v1.0 (Q4_K_M GGUF)..."
wget -q -nc https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

echo "Applying SadTalker compatibility patches..."
python3 src/sadtalker_patches.py

echo "Model setup complete."
