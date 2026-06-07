#!/usr/bin/env bash
# Localia RAG — lanceur macOS / Linux
set -e
cd "$(dirname "$0")"

echo "===================================="
echo "   Localia RAG - demarrage"
echo "===================================="

if ! command -v python3 >/dev/null 2>&1; then
  echo "[X] Python 3 introuvable. Installe-le depuis https://www.python.org/downloads/"
  exit 1
fi

if [ ! -x ".venv/bin/python" ]; then
  echo "[*] Premiere utilisation : installation de l'environnement (1-2 min)..."
  python3 -m venv .venv
  ./.venv/bin/python -m pip install --upgrade pip
  ./.venv/bin/python -m pip install -r requirements.txt
fi

if ! command -v ollama >/dev/null 2>&1; then
  echo "[X] Ollama introuvable. Installe-le depuis https://ollama.com puis relance."
  exit 1
fi

echo "[*] Verification des modeles (telechargement possible la 1re fois)..."
ollama pull nomic-embed-text
ollama pull qwen2.5:7b

echo "[*] Lancement de Localia RAG... une page va s'ouvrir dans ton navigateur."
./.venv/bin/python app.py
