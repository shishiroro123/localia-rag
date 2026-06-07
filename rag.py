"""
localia-rag — a minimal local RAG engine on top of Ollama.

No heavy dependencies: requests + numpy (+ pypdf for PDFs).
Everything runs locally. Nothing is sent to the internet.

An independent resource by Localia (https://getlocalia.com) — MIT.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import requests

# --- Configuration (override with environment variables) ---------------------

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
LLM_MODEL = os.environ.get("LOCALIA_RAG_MODEL", "qwen2.5:7b")
EMBED_MODEL = os.environ.get("LOCALIA_RAG_EMBED_MODEL", "nomic-embed-text")

CHUNK_WORDS = int(os.environ.get("LOCALIA_RAG_CHUNK_WORDS", "180"))
CHUNK_OVERLAP = int(os.environ.get("LOCALIA_RAG_CHUNK_OVERLAP", "40"))
TOP_K = int(os.environ.get("LOCALIA_RAG_TOP_K", "4"))

SUPPORTED_EXT = {".txt", ".md", ".markdown", ".pdf"}

SYSTEM_PROMPT = (
    "You are a precise local assistant answering questions about the user's own "
    "documents. Use ONLY the information in the CONTEXT below. If the answer is "
    "not in the context, say clearly that you cannot find it in the documents — "
    "never invent facts. Quote the source file name when relevant. Answer in the "
    "same language as the question, concisely."
)


# --- Ollama client -----------------------------------------------------------

class OllamaError(RuntimeError):
    """A clear, actionable error for the end user (not a stack trace)."""


def _post(path: str, payload: dict, stream: bool = False, timeout: int = 600):
    url = f"{OLLAMA_HOST}{path}"
    try:
        return requests.post(url, json=payload, stream=stream, timeout=timeout)
    except requests.exceptions.ConnectionError as exc:
        raise OllamaError(
            f"Cannot reach Ollama at {OLLAMA_HOST}.\n"
            "-> Is Ollama running? Start it (the Ollama app, or run 'ollama serve').\n"
            "-> Not installed? Get it at https://ollama.com"
        ) from exc


def ollama_up() -> bool:
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        return r.status_code == 200
    except requests.exceptions.RequestException:
        return False


def list_models() -> list[str]:
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=10)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except requests.exceptions.RequestException as exc:
        raise OllamaError(f"Could not read Ollama models: {exc}") from exc


def _model_present(model: str, available: list[str]) -> bool:
    # "qwen2.5:7b" should match "qwen2.5:7b" and the ":latest" variant
    base = model.split(":")[0]
    return any(m == model or m.split(":")[0] == base for m in available)


def check_environment() -> list[str]:
    """Return a list of problems (empty list = everything is fine)."""
    problems: list[str] = []
    if not ollama_up():
        problems.append(
            f"Ollama is not reachable at {OLLAMA_HOST}. Start Ollama (https://ollama.com)."
        )
        return problems
    available = list_models()
    if not _model_present(EMBED_MODEL, available):
        problems.append(f"Missing embedding model: '{EMBED_MODEL}'. "
                        f"Install it: ollama pull {EMBED_MODEL}")
    if not _model_present(LLM_MODEL, available):
        problems.append(f"Missing chat model: '{LLM_MODEL}'. "
                        f"Install it: ollama pull {LLM_MODEL}")
    return problems


# --- Embeddings --------------------------------------------------------------

def embed_texts(texts: list[str]) -> np.ndarray:
    """Normalized embeddings (N x D). Works with /api/embed (recent) and
    /api/embeddings (older Ollama)."""
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    # Recent API: batched
    resp = _post("/api/embed", {"model": EMBED_MODEL, "input": texts})
    if resp.status_code == 200:
        data = resp.json()
        vecs = data.get("embeddings")
        if vecs:
            return _normalize(np.asarray(vecs, dtype=np.float32))

    # Fallback to the older API: one call per text
    out = []
    for t in texts:
        r = _post("/api/embeddings", {"model": EMBED_MODEL, "prompt": t})
        if r.status_code != 200:
            raise OllamaError(
                f"Embeddings failed (model '{EMBED_MODEL}'). "
                f"Ollama response: {r.status_code} {r.text[:200]}"
            )
        out.append(r.json()["embedding"])
    return _normalize(np.asarray(out, dtype=np.float32))


def embed_one(text: str) -> np.ndarray:
    return embed_texts([text])[0]


def _normalize(mat: np.ndarray) -> np.ndarray:
    if mat.size == 0:
        return mat
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mat / norms).astype(np.float32)


# --- Chat --------------------------------------------------------------------

def chat_stream(messages: list[dict], model: str | None = None) -> Iterator[str]:
    """Generate the answer token by token (Ollama NDJSON streaming)."""
    payload = {"model": model or LLM_MODEL, "messages": messages, "stream": True}
    resp = _post("/api/chat", payload, stream=True)
    if resp.status_code != 200:
        raise OllamaError(f"Chat failed: {resp.status_code} {resp.text[:200]}")
    for line in resp.iter_lines():
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("done"):
            break
        piece = obj.get("message", {}).get("content", "")
        if piece:
            yield piece


def chat(messages: list[dict], model: str | None = None) -> str:
    return "".join(chat_stream(messages, model))


# --- Reading documents -------------------------------------------------------

def read_file(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise OllamaError(
                "Reading PDFs requires pypdf. Install it: pip install pypdf"
            ) from exc
        reader = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    # txt / md / markdown
    return path.read_text(encoding="utf-8", errors="ignore")


def iter_documents(folder: str | Path) -> Iterator[tuple[str, str]]:
    folder = Path(folder)
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXT:
            text = read_file(path).strip()
            if text:
                yield (str(path.relative_to(folder)), text)


# --- Chunking ----------------------------------------------------------------

def chunk_text(text: str, size: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    if not words:
        return []
    if len(words) <= size:
        return [" ".join(words)]
    step = max(1, size - overlap)
    chunks = []
    for start in range(0, len(words), step):
        chunk = words[start:start + size]
        if chunk:
            chunks.append(" ".join(chunk))
        if start + size >= len(words):
            break
    return chunks


# --- Vector index ------------------------------------------------------------

@dataclass
class VectorIndex:
    chunks: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    embeddings: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))

    def __len__(self) -> int:
        return len(self.chunks)

    def add(self, chunks: list[str], sources: list[str], embeddings: np.ndarray) -> None:
        if not chunks:
            return
        self.chunks.extend(chunks)
        self.sources.extend(sources)
        if self.embeddings.size == 0:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])

    def search(self, query_emb: np.ndarray, k: int = TOP_K) -> list[tuple[float, str, str]]:
        if len(self) == 0:
            return []
        scores = self.embeddings @ query_emb  # cosine (vectors are normalized)
        k = min(k, len(self))
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [(float(scores[i]), self.chunks[i], self.sources[i]) for i in idx]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path.with_suffix(".npy"), self.embeddings)
        path.with_suffix(".json").write_text(
            json.dumps({"chunks": self.chunks, "sources": self.sources}, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "VectorIndex":
        path = Path(path)
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        emb = np.load(path.with_suffix(".npy"))
        return cls(chunks=meta["chunks"], sources=meta["sources"], embeddings=emb)

    @classmethod
    def exists(cls, path: str | Path) -> bool:
        path = Path(path)
        return path.with_suffix(".json").exists() and path.with_suffix(".npy").exists()


def build_index(folder: str | Path, on_progress=None) -> VectorIndex:
    """Build the index from a folder of documents. Optional on_progress(msg)."""
    index = VectorIndex()
    docs = list(iter_documents(folder))
    if not docs:
        return index
    for i, (source, text) in enumerate(docs, 1):
        pieces = chunk_text(text)
        if not pieces:
            continue
        if on_progress:
            on_progress(f"[{i}/{len(docs)}] {source} - {len(pieces)} chunks")
        embs = embed_texts(pieces)
        index.add(pieces, [source] * len(pieces), embs)
    return index


# --- RAG pipeline ------------------------------------------------------------

def build_messages(question: str, hits: list[tuple[float, str, str]],
                   history: list[dict] | None = None) -> list[dict]:
    if hits:
        context = "\n\n".join(
            f"[Source: {src}]\n{chunk}" for _, chunk, src in hits
        )
    else:
        context = "(no relevant passage found in the documents)"
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({
        "role": "user",
        "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}",
    })
    return messages


def answer_stream(question: str, index: VectorIndex,
                  history: list[dict] | None = None,
                  k: int = TOP_K) -> tuple[Iterator[str], list[str]]:
    """Return (token generator, list of source files used)."""
    if len(index) == 0:
        def _empty():
            yield ("No documents indexed yet. Add files to the 'documents/' "
                   "folder, then (re)build the index.")
        return _empty(), []
    q_emb = embed_one(question)
    hits = index.search(q_emb, k)
    sources = sorted({src for _, _, src in hits})
    messages = build_messages(question, hits, history)
    return chat_stream(messages), sources


if __name__ == "__main__":
    # Tiny self-test without Ollama (pure logic)
    print("chunk_text:", len(chunk_text(" ".join(["word"] * 500))), "chunks")
    idx = VectorIndex()
    idx.add(["a", "b"], ["s1", "s1"], _normalize(np.array([[1, 0], [0, 1]], dtype=np.float32)))
    print("search:", idx.search(np.array([1, 0], dtype=np.float32), k=1))
    print("OK")
