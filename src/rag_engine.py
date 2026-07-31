# ================================================================
#  rag_engine.py  —  RAG brain for TU BSc CSIT Study Buddy
#  Pipeline: query → hybrid retrieve → rerank → LLM → answer
#  LLM: Ollama (default) or vllm — both via OpenAI-compatible API
# ================================================================

import os
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from flashrank import Ranker, RerankRequest
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env", override=False)

EMBED_MODEL    = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
LLM_BASE_URL   = os.getenv("LLM_BASE_URL",   "http://localhost:11434/v1")
LLM_API_KEY    = os.getenv("LLM_API_KEY",    "ollama")
LLM_MODEL      = os.getenv("LLM_MODEL",      "qwen2.5:7b")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", 900))
TOP_K          = int(os.getenv("TOP_K",       10))
RERANK_TOP     = int(os.getenv("RERANK_TOP",   4))

# ── Lazy-loaded singletons — initialized on first query ──────────
_embedder = None
_chroma   = None
_reranker = None
_llm      = None


def _get_resources():
    global _embedder, _chroma, _reranker, _llm
    if _llm is not None:
        return

    from src.chroma_client import get_chroma_client

    print("Loading embedding model...")
    _embedder = SentenceTransformer(EMBED_MODEL)

    print("Connecting to ChromaDB...")
    _chroma = get_chroma_client()

    print("Loading re-ranker...")
    _reranker = Ranker(
        model_name="ms-marco-MiniLM-L-12-v2",
        cache_dir=str(BASE_DIR / "VectorStore" / "reranker_cache"),
    )

    print(f"Connecting to LLM at {LLM_BASE_URL} (model: {LLM_MODEL})...")
    _llm = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
    print("RAG engine ready.\n")


# ── Collection discovery ─────────────────────────────────────────
def _collections(sem=None, subject=None):
    all_names = [c.name for c in _chroma.list_collections()]
    if sem and subject:
        target = f"sem{sem}_{subject}"
        return [_chroma.get_collection(target)] if target in all_names else []
    if sem:
        return [_chroma.get_collection(n) for n in all_names if n.startswith(f"sem{sem}_")]
    return [_chroma.get_collection(n) for n in all_names]


# ── Vector search ────────────────────────────────────────────────
def _vector_search(query: str, cols: list) -> list[dict]:
    if not cols:
        return []
    qvec    = _embedder.encode(query).tolist()
    results = []
    for col in cols:
        if col.count() == 0:
            continue
        try:
            res = col.query(
                query_embeddings=[qvec],
                n_results=min(TOP_K, col.count()),
                include=["documents", "metadatas", "distances"],
            )
            for doc, meta, dist in zip(
                res["documents"][0], res["metadatas"][0], res["distances"][0]
            ):
                results.append({
                    "text":    doc,
                    "file":    meta.get("file",    ""),
                    "page":    meta.get("page",    "?"),
                    "sem":     meta.get("sem",     "?"),
                    "subject": meta.get("subject", "?"),
                    "score":   round(1 - dist, 4),
                })
        except Exception:
            continue
    return results


# ── BM25 keyword search ──────────────────────────────────────────
def _bm25_search(query: str, cols: list) -> list[dict]:
    all_docs, all_metas = [], []
    for col in cols:
        if col.count() == 0:
            continue
        data = col.get(include=["documents", "metadatas"])
        all_docs.extend(data["documents"])
        all_metas.extend(data["metadatas"])
    if not all_docs:
        return []
    tokenized = [d.lower().split() for d in all_docs]
    if not any(tokenized):
        return []
    bm25   = BM25Okapi(tokenized)
    scores = bm25.get_scores(query.lower().split())
    top    = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:TOP_K]
    return [{
        "text":    all_docs[i],
        "file":    all_metas[i].get("file",    ""),
        "page":    all_metas[i].get("page",    "?"),
        "sem":     all_metas[i].get("sem",     "?"),
        "subject": all_metas[i].get("subject", "?"),
        "score":   round(float(scores[i]), 4),
    } for i in top if scores[i] > 0]


# ── Merge + dedup ────────────────────────────────────────────────
def _hybrid(query: str, cols: list) -> list[dict]:
    seen, merged = set(), []
    for c in _vector_search(query, cols) + _bm25_search(query, cols):
        key = c["text"][:120]
        if key not in seen:
            seen.add(key)
            merged.append(c)
    return merged


# ── Re-rank ──────────────────────────────────────────────────────
def _rerank(query: str, chunks: list[dict]) -> list[dict]:
    if not chunks:
        return []
    passages = [{"id": i, "text": c["text"]} for i, c in enumerate(chunks)]
    ranked   = _reranker.rerank(RerankRequest(query=query, passages=passages))
    top      = sorted(ranked, key=lambda x: x["score"], reverse=True)[:RERANK_TOP]
    return [chunks[r["id"]] for r in top]


# ── Build prompt ─────────────────────────────────────────────────
def _prompt(query: str, chunks: list[dict], history: list) -> list[dict]:
    context = ""
    for i, c in enumerate(chunks, 1):
        context += (
            f"\n[Source {i} | Sem {c['sem']} | {c['subject']} "
            f"| {c['file']} | Chapter/Page {c['page']}]\n{c['text']}\n"
        )

    system = f"""You are AI Study Buddy for TU BSc CSIT students in Nepal.
You help students understand their exact course material clearly.

RULES:
1. Answer ONLY using the context provided. Do not use outside knowledge.
2. Always cite the source file and chapter/page number.
3. If the answer is not in the context say:
   "I couldn't find this in your uploaded material. Please check your textbook or ask your teacher."
4. Keep explanations simple and clear. Use examples where helpful.
5. Never make up definitions, formulas, or facts.

CONTEXT:
{context}"""

    messages = [{"role": "system", "content": system}]
    for turn in history[-6:]:
        messages.append({"role": "user",      "content": turn["question"]})
        messages.append({"role": "assistant", "content": turn["answer"]})
    messages.append({"role": "user", "content": query})
    return messages


# ── Generate answer ──────────────────────────────────────────────
def _generate(messages: list[dict]) -> str:
    try:
        response = _llm.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.1,
            max_tokens=LLM_MAX_TOKENS,
        )
        return response.choices[0].message.content
    except Exception as exc:
        return (
            f"Could not reach the local LLM. "
            f"Make sure Ollama is running (`ollama serve`) and the model is pulled "
            f"(`ollama pull {LLM_MODEL}`). "
            f"Error: {type(exc).__name__}: {exc}"
        )


# ── Public entry point ───────────────────────────────────────────
def get_answer(
    query: str,
    sem: str = None,
    subject: str = None,
    history: list = None,
) -> dict:
    _get_resources()
    if history is None:
        history = []

    cols = _collections(sem, subject)
    if not cols:
        return {
            "answer": (
                "No study material found. Run the ingest script first:\n\n"
                "`uv run python src/ingest.py --sem all`"
            ),
            "sources": [],
        }

    chunks     = _hybrid(query, cols)
    top_chunks = _rerank(query, chunks)

    if not top_chunks:
        return {
            "answer":  "I couldn't find relevant content for this question in your study material.",
            "sources": [],
        }

    seen, sources = set(), []
    for c in top_chunks:
        key = f"{c['file']}_{c['page']}"
        if key not in seen:
            seen.add(key)
            sources.append({
                "file":    c["file"],
                "page":    c["page"],
                "sem":     c["sem"],
                "subject": c["subject"],
            })

    return {
        "answer":  _generate(_prompt(query, top_chunks, history)),
        "sources": sources,
    }
