"""Extract local EPUB/PDF books, chunk them, and index one Chroma collection."""

import argparse
import os
import re
from pathlib import Path

import ebooklib
import fitz
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from ebooklib import epub
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

try:
    from src.catalog import available_subjects
    from src.chroma_client import get_chroma_client
except ModuleNotFoundError:
    from catalog import available_subjects
    from chroma_client import get_chroma_client


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env", override=False)

EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "google/embeddinggemma-300m")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 2200))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 320))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "study_material")


def extract_epub(path: Path, sem: str, subject: str) -> list[dict]:
    try:
        book, documents, number = epub.read_epub(str(path)), [], 1
        for item in book.get_items():
            if item.get_type() != ebooklib.ITEM_DOCUMENT:
                continue
            soup = BeautifulSoup(item.get_content(), "html.parser")
            for tag in soup(["script", "style", "head"]):
                tag.decompose()
            blocks = soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "blockquote"])
            text = "\n\n".join(block.get_text(" ", strip=True) for block in blocks).strip()
            if not text:
                text = soup.get_text(separator="\n\n").strip()
            if len(text) >= 100:
                documents.append({"text": text, "page": number, "file": path.name,
                                  "sem": sem, "subject": subject})
                number += 1
        print(f"       ✓ {path.name} → {len(documents)} EPUB sections")
        return documents
    except Exception as exc:
        print(f"       ❌ {path.name} failed: {exc}")
        return []


def extract_pdf(path: Path, sem: str, subject: str) -> list[dict]:
    try:
        document, pages = fitz.open(str(path)), []
        for index, page in enumerate(document):
            text = page.get_text("text").strip()
            if len(text) >= 60:
                pages.append({"text": text, "page": index + 1, "file": path.name,
                              "sem": sem, "subject": subject})
        document.close()
        print(f"       ✓ {path.name} → {len(pages)} PDF pages")
        return pages
    except Exception as exc:
        print(f"       ❌ {path.name} failed: {exc}")
        return []


def _clean(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE).strip()


def _split_long_paragraph(paragraph: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    pieces, current = [], ""
    for sentence in sentences:
        words = sentence.split()
        if not words:
            continue
        sentence = " ".join(words)
        if current and len(current) + len(sentence) + 1 > CHUNK_SIZE:
            pieces.append(current)
            current = ""
        if len(sentence) <= CHUNK_SIZE:
            current = f"{current} {sentence}".strip()
            continue
        for word in words:
            if current and len(current) + len(word) + 1 > CHUNK_SIZE:
                pieces.append(current)
                current = ""
            current = f"{current} {word}".strip()
    if current:
        pieces.append(current)
    return pieces


def _tail(paragraphs: list[str]) -> list[str]:
    overlap, size = [], 0
    for paragraph in reversed(paragraphs):
        if overlap and size + len(paragraph) > CHUNK_OVERLAP:
            break
        overlap.insert(0, paragraph)
        size += len(paragraph) + 2
    return overlap


def chunk(documents: list[dict]) -> list[dict]:
    """Chunk within a single PDF page or EPUB section; never across sources."""
    chunks, index = [], 0
    for document in documents:
        paragraphs = []
        for paragraph in re.split(r"\n\s*\n", _clean(document["text"])):
            paragraph = " ".join(paragraph.split())
            if paragraph:
                paragraphs.extend(_split_long_paragraph(paragraph))

        current = []
        for paragraph in paragraphs:
            if current and sum(len(item) + 2 for item in current) + len(paragraph) > CHUNK_SIZE:
                chunks.append({**document, "id": f"s{document['sem']}_{document['subject']}_{document['file']}_p{document['page']}_c{index}",
                               "text": "\n\n".join(current)})
                index += 1
                current = _tail(current)
            current.append(paragraph)
        if current:
            text = "\n\n".join(current)
            if len(text) > 80:
                chunks.append({**document, "id": f"s{document['sem']}_{document['subject']}_{document['file']}_p{document['page']}_c{index}", "text": text})
                index += 1
    return chunks


def store(chunks: list[dict], embedder) -> None:
    if not chunks:
        print("       ⚠ No chunks to store")
        return
    collection = get_chroma_client().get_or_create_collection(
        COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )
    ids = [chunk["id"] for chunk in chunks]
    existing = set(collection.get(ids=ids)["ids"])
    new_chunks = [chunk for chunk in chunks if chunk["id"] not in existing]
    if not new_chunks:
        print(f"       ↩ {COLLECTION_NAME}: already indexed")
        return

    texts = [chunk["text"] for chunk in new_chunks]
    ids = [chunk["id"] for chunk in new_chunks]
    metadata = [{key: chunk[key] for key in ("file", "page", "sem", "subject")} for chunk in new_chunks]
    for start in tqdm(range(0, len(texts), 64), desc=f"       {COLLECTION_NAME}"):
        embeddings = embedder.encode(texts[start:start + 64], show_progress_bar=False).tolist()
        collection.upsert(embeddings=embeddings, documents=texts[start:start + 64],
                          ids=ids[start:start + 64], metadatas=metadata[start:start + 64])
    print(f"       ✅ {COLLECTION_NAME}: {len(new_chunks)} chunks stored (total: {collection.count()})")


def run(sem_arg: str, subject_arg: str | None = None) -> None:
    print(f"\nLoading embedding model: {EMBED_MODEL}")
    embedder = SentenceTransformer(EMBED_MODEL)
    records = available_subjects()
    sems = sorted({record["sem"] for record in records}) if sem_arg == "all" else [sem_arg]
    for sem in sems:
        matches = [record for record in records if record["sem"] == sem and
                   (subject_arg is None or record["slug"] == subject_arg)]
        if not matches:
            print(f"⚠ No matching EPUB/PDF folders found for semester {sem}")
            continue
        for record in matches:
            print(f"\n📚 {record['label']} ({len(record['files'])} file(s))")
            documents = []
            for path in record["files"]:
                extractor = extract_epub if path.suffix.lower() == ".epub" else extract_pdf
                documents.extend(extractor(path, sem, record["slug"]))
            chunks = chunk(documents)
            print(f"       → {len(chunks)} structured chunks")
            store(chunks, embedder)
    print("\n✅ Done! Run: uv run streamlit run app.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sem", required=True)
    parser.add_argument("--subject")
    args = parser.parse_args()
    run(args.sem, args.subject)
