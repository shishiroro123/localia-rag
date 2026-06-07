"""
localia-rag — a local chat UI to talk with your own documents.

Just run:  python app.py
A chat window opens in your browser. Everything stays on your machine.

Options:
  python app.py            # graphical interface (default)
  python app.py --check     # diagnose the environment
  python app.py --reindex   # (re)index the documents/ folder, then exit
  python app.py --cli       # chat in the terminal, no browser

An independent resource by Localia (https://getlocalia.com) — MIT.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import rag

# Force UTF-8 console on Windows (emojis crash the default cp1252 codec).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

DOCS_DIR = Path("documents")
INDEX_PATH = Path(".index/store")

INDEX: rag.VectorIndex | None = None


# --- Index -------------------------------------------------------------------

def load_or_build_index() -> rag.VectorIndex:
    global INDEX
    if rag.VectorIndex.exists(INDEX_PATH):
        INDEX = rag.VectorIndex.load(INDEX_PATH)
    else:
        DOCS_DIR.mkdir(exist_ok=True)
        INDEX = rag.build_index(DOCS_DIR, on_progress=lambda m: print("  ", m))
        if len(INDEX):
            INDEX.save(INDEX_PATH)
    return INDEX


def index_status() -> str:
    n = len(INDEX) if INDEX else 0
    if n == 0:
        return ("⚠️ No documents indexed yet. Drop files into the **documents/** "
                "folder (or use the button below), then click *Re-index*.")
    files = len(set(INDEX.sources))
    return f"✅ Index ready: **{n} chunks** from **{files} file(s)**."


def reindex() -> str:
    global INDEX
    try:
        DOCS_DIR.mkdir(exist_ok=True)
        INDEX = rag.build_index(DOCS_DIR, on_progress=lambda m: print("  ", m))
        if len(INDEX):
            INDEX.save(INDEX_PATH)
        return index_status()
    except rag.OllamaError as exc:
        return f"❌ {exc}"


def add_files(files) -> str:
    if not files:
        return index_status()
    DOCS_DIR.mkdir(exist_ok=True)
    added = 0
    for f in files:
        src = Path(f)
        if src.suffix.lower() in rag.SUPPORTED_EXT:
            shutil.copy(src, DOCS_DIR / src.name)
            added += 1
    if added == 0:
        return "❌ No compatible file (.txt, .md, .pdf)."
    return reindex()


# --- Graphical interface (Gradio) -------------------------------------------

def respond(message: str, history: list[dict]):
    message = (message or "").strip()
    if not message:
        yield "", history
        return
    history = history + [{"role": "user", "content": message},
                         {"role": "assistant", "content": ""}]
    try:
        stream, sources = rag.answer_stream(message, INDEX)
        for piece in stream:
            history[-1]["content"] += piece
            yield "", history
        if sources:
            history[-1]["content"] += f"\n\n— 📄 *{', '.join(sources)}*"
            yield "", history
    except rag.OllamaError as exc:
        history[-1]["content"] = f"❌ {exc}"
        yield "", history


def build_ui():
    import gradio as gr

    with gr.Blocks(title="Localia RAG") as demo:
        gr.Markdown(
            "# 🧠 Localia RAG\n"
            "Chat with your own documents, **100% locally** (on top of Ollama). "
            "Nothing leaves your machine."
        )
        status = gr.Markdown(index_status())
        chatbot = gr.Chatbot(height=460, show_label=False)
        with gr.Row():
            msg = gr.Textbox(placeholder="Ask a question about your documents…",
                             scale=8, show_label=False, autofocus=True)
            send = gr.Button("Send", variant="primary", scale=1, min_width=110)

        with gr.Accordion("📁 My documents", open=False):
            gr.Markdown(
                "Add **.txt, .md or .pdf** files. They are copied into the "
                "`documents/` folder and indexed locally."
            )
            uploader = gr.File(file_count="multiple", label="Add files",
                               file_types=[".txt", ".md", ".markdown", ".pdf"])
            reindex_btn = gr.Button("🔄 Re-index the documents/ folder")

        gr.Markdown(
            "<sub>Localia RAG — independent resource · "
            "[getlocalia.com](https://getlocalia.com) · no data sent online</sub>"
        )

        msg.submit(respond, [msg, chatbot], [msg, chatbot])
        send.click(respond, [msg, chatbot], [msg, chatbot])
        uploader.upload(add_files, uploader, status)
        reindex_btn.click(lambda: "⏳ Indexing…", None, status).then(
            reindex, None, status)

    return demo


# --- Terminal mode -----------------------------------------------------------

def cli_loop() -> None:
    print("\nLocalia RAG — terminal chat. Type your question (Ctrl+C to quit).\n")
    while True:
        try:
            question = input("You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        try:
            stream, sources = rag.answer_stream(question, INDEX)
            print("RAG > ", end="", flush=True)
            for piece in stream:
                print(piece, end="", flush=True)
            if sources:
                print(f"\n      📄 {', '.join(sources)}")
            print()
        except rag.OllamaError as exc:
            print(f"\n❌ {exc}\n")


# --- Entry point -------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Localia RAG — a simple local RAG on top of Ollama.")
    ap.add_argument("--check", action="store_true",
                    help="Diagnose the environment (Ollama, models) and exit.")
    ap.add_argument("--reindex", action="store_true",
                    help="(Re)index the documents/ folder and exit.")
    ap.add_argument("--cli", action="store_true",
                    help="Chat in the terminal, without the web UI.")
    ap.add_argument("--port", type=int, default=7860, help="Web UI port.")
    ap.add_argument("--share", action="store_true",
                    help="Create a temporary public link (not recommended for private docs).")
    args = ap.parse_args()

    print(f"Ollama: {rag.OLLAMA_HOST}  |  LLM: {rag.LLM_MODEL}  |  "
          f"embeddings: {rag.EMBED_MODEL}")

    problems = rag.check_environment()
    if args.check:
        if problems:
            print("\nProblems found:")
            for p in problems:
                print("  -", p)
            sys.exit(1)
        print("✅ All set (Ollama reachable, models present).")
        return

    if problems:
        print("\n⚠️  Heads up:")
        for p in problems:
            print("  -", p)
        print("   Continuing anyway, but fix this for it to work.\n")

    print("Loading index…")
    load_or_build_index()
    print(index_status().replace("**", ""))

    if args.reindex:
        print(reindex().replace("**", ""))
        return
    if args.cli:
        cli_loop()
        return

    demo = build_ui()
    print(f"\n🧠 Localia RAG is starting on http://127.0.0.1:{args.port}\n")
    demo.launch(server_name="127.0.0.1", server_port=args.port,
                share=args.share, inbrowser=True)


if __name__ == "__main__":
    main()
