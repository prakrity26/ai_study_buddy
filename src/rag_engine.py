# ================================================================
#  rag_engine.py  —  RAG brain for TU BSc CSIT Study Buddy
#  Pipeline: query → hybrid retrieve → rerank → LLM → answer
# ================================================================

import os
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from flashrank import Ranker, RerankRequest
from src.chroma_client import get_chroma_client
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env", override=False)

EMBED_MODEL        = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
LLM_MODEL          = os.getenv("LLM_MODEL", "Qwen/Qwen3.5-4B")
TOP_K              = int(os.getenv("TOP_K", 10))
RERANK_TOP         = int(os.getenv("RERANK_TOP", 4))
LLM_MAX_NEW_TOKENS = int(os.getenv("LLM_MAX_NEW_TOKENS", 900))
COLLECTION_NAME    = os.getenv("CHROMA_COLLECTION", "study_material")

print("Loading embedding model...")
embedder = SentenceTransformer(EMBED_MODEL)

print("Connecting to ChromaDB...")
chroma = get_chroma_client()

print("Loading re-ranker...")
reranker = Ranker(
    model_name="ms-marco-MiniLM-L-12-v2",
    cache_dir=str(BASE_DIR / "VectorStore" / "reranker_cache")
)

print(f"Loading local Hugging Face LLM: {LLM_MODEL}...")
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
llm = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL,
    torch_dtype="auto",
    device_map="auto",
)
print("✅ RAG engine ready\n")


def _collection():
    """The whole library lives in one collection; filters live in metadata."""
    try:
        return chroma.get_collection(COLLECTION_NAME)
    except Exception:
        return None


def _where(sem: str | None, subject: str | None) -> dict | None:
    filters = []
    if sem:
        filters.append({"sem": sem})
    if subject:
        filters.append({"subject": subject})
    if not filters:
        return None
    return filters[0] if len(filters) == 1 else {"$and": filters}


# ── Vector search ────────────────────────────────────────────────
def _vector_search(query: str, collection, where: dict | None) -> list[dict]:
    if not collection or collection.count() == 0:
        return []
    qvec = embedder.encode(query).tolist()
    results = []
    try:
        kwargs = {"query_embeddings": [qvec], "n_results": TOP_K,
                  "include": ["documents", "metadatas", "distances"]}
        if where:
            kwargs["where"] = where
        res = collection.query(**kwargs)
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            results.append({"text": doc, "file": meta.get("file", ""),
                            "page": meta.get("page", "?"), "sem": meta.get("sem", "?"),
                            "subject": meta.get("subject", "?"), "score": round(1 - dist, 4)})
    except Exception:
        return []
    return results


# ── BM25 keyword search ──────────────────────────────────────────
def _bm25_search(query: str, collection, where: dict | None) -> list[dict]:
    if not collection or collection.count() == 0:
        return []
    kwargs = {"include": ["documents", "metadatas"]}
    if where:
        kwargs["where"] = where
    data = collection.get(**kwargs)
    all_docs, all_metas = data["documents"], data["metadatas"]
    if not all_docs:
        return []
    bm25   = BM25Okapi([d.lower().split() for d in all_docs])
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
def _hybrid(query: str, collection, where: dict | None) -> list[dict]:
    seen, merged = set(), []
    for c in _vector_search(query, collection, where) + _bm25_search(query, collection, where):
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
    ranked   = reranker.rerank(RerankRequest(query=query, passages=passages))
    top      = sorted(ranked, key=lambda x: x["score"], reverse=True)[:RERANK_TOP]
    return [chunks[r["id"]] for r in top]


# ── Build prompt ─────────────────────────────────────────────────
def _prompt(query: str, chunks: list[dict], history: list) -> list[dict]:
    context = ""
    for i, c in enumerate(chunks, 1):
        context += f"\n[Source {i} | Sem {c['sem']} | {c['subject']} | {c['file']} | Chapter/Page {c['page']}]\n{c['text']}\n"

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

def _generate(messages: list[dict]) -> str:
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer(text, return_tensors="pt").to(llm.device)

    output_ids = llm.generate(
        **inputs,
        max_new_tokens=400,
        do_sample=True,
        temperature=0.7,
        top_p=0.8,
    )

    generated_ids = output_ids[:, inputs.input_ids.shape[-1]:]
    return tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )[0].strip()

def get_answer(
    query: str,
    sem: str = None,
    subject: str = None,
    history: list = None,
) -> dict:
    if history is None:
        history = []

    collection = _collection()
    if not collection or collection.count() == 0:
        return {
            "answer": "No study material found. Please ingest the folders in Data first.",
            "sources": [],
        }

    chunks = _hybrid(query, collection, _where(sem, subject))
    top_chunks = _rerank(query, chunks)

    if not top_chunks:
        return {
            "answer": "I couldn't find relevant content for this question in your study material.",
            "sources": [],
        }

    seen, sources = set(), []
    for chunk in top_chunks:
        key = f"{chunk['file']}_{chunk['page']}"
        if key not in seen:
            seen.add(key)
            sources.append({
                "file": chunk["file"],
                "page": chunk["page"],
                "sem": chunk["sem"],
                "subject": chunk["subject"],
            })

    return {
        "answer": _generate(_prompt(query, top_chunks, history)),
        "sources": sources,
    }