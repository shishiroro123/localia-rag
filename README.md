# 🧠 Localia RAG

**Chat with your own documents, 100% locally — as simple as Ollama.**

Drop your files in a folder, run one command, and a chat window opens in your
browser. Ask questions, get answers grounded in *your* documents, with the
source file cited. Nothing ever leaves your machine.

No cloud. No API key. No account. Free and open source.

```
You > What is the Atlas project budget and who is the manager?
RAG > The Atlas project budget is 48,500 euros, and the project manager
      is Camille Renard.
      📄 example.md
```

---

## Why

Most "chat with your PDF" tools send your documents to a cloud API. Localia RAG
does the opposite: it runs entirely on your computer, on top of
[Ollama](https://ollama.com). Your data stays private, it's free, and it works
offline.

- 🔒 **Private** — documents and answers never leave your machine.
- ⚡ **Simple** — one command, a browser chat window opens automatically.
- 🧩 **Lightweight** — pure Python, no database to install, no Docker required.
- 📄 **Grounded** — answers cite the source file; it says "I don't know" instead
  of making things up.
- 🌍 **Any language** — it answers in the language you ask in.

## Requirements

1. **[Ollama](https://ollama.com)** installed (it handles the AI models and your GPU).
2. **Python 3.10+** ([python.org](https://www.python.org/downloads/) — on Windows, tick *"Add python.exe to PATH"*).

That's it. The launcher installs everything else and downloads the models on first run.

## Quick start

**Windows** — double-click **`run.bat`**

**macOS / Linux**
```bash
chmod +x run.sh && ./run.sh
```

**Manual (any OS)**
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

The first run downloads two models via Ollama (~5 GB total, once):
`qwen2.5:7b` (the chat model) and `nomic-embed-text` (the embeddings). Then a
chat window opens at `http://127.0.0.1:7860`.

## How to use it

1. Put your `.txt`, `.md` or `.pdf` files in the **`documents/`** folder
   (or use the **My documents** button in the UI to add them).
2. Click **Re-index** (the first launch indexes them automatically).
3. Ask questions in the chat.

### Terminal mode (no browser)
```bash
python app.py --cli
```

### Check your setup
```bash
python app.py --check
```

## Configuration

Everything is optional and set through environment variables:

| Variable | Default | What it does |
|---|---|---|
| `LOCALIA_RAG_MODEL` | `qwen2.5:7b` | The chat model. Lighter option: `llama3.2:3b`. Better: `qwen2.5:14b`. |
| `LOCALIA_RAG_EMBED_MODEL` | `nomic-embed-text` | The embedding model. |
| `OLLAMA_HOST` | `http://localhost:11434` | Where Ollama runs. |
| `LOCALIA_RAG_TOP_K` | `4` | How many passages to retrieve per question. |

Example (use a smaller, faster model):
```bash
# Windows:  set LOCALIA_RAG_MODEL=llama3.2:3b
# macOS/Linux:
LOCALIA_RAG_MODEL=llama3.2:3b python app.py
```

## How it works

A classic, minimal RAG pipeline — kept deliberately small and readable:

1. **Ingest** — your documents are split into overlapping word chunks (`rag.py`).
2. **Embed** — each chunk is turned into a vector with Ollama's embedding model.
3. **Store** — vectors live in a plain NumPy array on disk (`.index/`). No
   database, no native dependency.
4. **Retrieve** — your question is embedded and compared (cosine similarity) to
   find the most relevant chunks.
5. **Answer** — those chunks are handed to the chat model with a strict
   instruction: *answer only from this context, and cite the source.*

The whole engine is ~320 lines in [`rag.py`](rag.py). Read it, fork it, tweak it.

## Privacy

Localia RAG makes **no outbound network calls** except to your local Ollama
instance. Your documents, your index and your questions stay on your computer.
(The only exception: if you pass `--share`, Gradio creates a temporary public
tunnel — don't use it with private documents.)

## About

Built by **[Localia](https://getlocalia.com)** — an independent resource for
running AI locally. Not sure which GPU runs which model? Try the free
[GPU → LLM calculator](https://getlocalia.com/calculateur) (200+ GPUs, 240+ models).

We don't sell hardware. This tool is free and MIT-licensed — use it, share it,
build on it.

## License

[MIT](LICENSE) © Localia
