#!/usr/bin/env bash
# Localia RAG - launcher for macOS / Linux
set -e
cd "$(dirname "$0")"

echo "===================================="
echo "   Localia RAG - starting"
echo "===================================="

if ! command -v python3 >/dev/null 2>&1; then
  echo "[X] Python 3 not found. Install it from https://www.python.org/downloads/"
  exit 1
fi

if [ ! -x ".venv/bin/python" ]; then
  echo "[*] First run: setting up the environment (1-2 min)..."
  python3 -m venv .venv
  ./.venv/bin/python -m pip install --upgrade pip
  ./.venv/bin/python -m pip install -r requirements.txt
fi

if ! command -v ollama >/dev/null 2>&1; then
  echo "[X] Ollama not found. Install it from https://ollama.com then run again."
  exit 1
fi

echo "[*] Checking models (a download may happen on first run)..."
ollama pull nomic-embed-text
ollama pull qwen2.5:7b

echo "[*] Starting Localia RAG... a page will open in your browser."
./.venv/bin/python app.py
