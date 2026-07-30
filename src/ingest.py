# ================================================================
#  ingest.py  —  TU BSc CSIT AI Study Buddy
#  Reads EPUB (or PDF) files → chunks → stores in ChromaDB
#
#  USAGE:
#    Single subject:  uv run python src/ingest.py --sem 4 --subject operating_systems
#    Full semester:   uv run python src/ingest.py --sem 4
#    Everything:      uv run python src/ingest.py --sem all
#
#  YOUR DATA FOLDER STRUCTURE:
#    Data/
#      Sem4/
#        operating_systems/
#          OS_Galvin.epub        ← place epub here
#        computer_networks/
#          CN_Forouzan.epub
# ================================================================

import os, re, argparse
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import fitz
from sentence_transformers import SentenceTransformer

try:
    from src.chroma_client import get_chroma_client
except ModuleNotFoundError:
    from chroma_client import get_chroma_client

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env", override=False)

EMBED_MODEL   = os.getenv("EMBEDDING_MODEL",   "google/embeddinggemma-300m")
CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE",    600))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 80))

# ── Your exact TU BSc CSIT subjects (skipping maths/stats/DSA) ──
SEMESTER_SUBJECTS = {
    "1": [
        "introduction_to_it",
        "c_programming",
        "digital_logic",
        "physics",
    ],
    "2": [
        "object_oriented_programming_cpp",
        "microprocessor",
    ],
    "3": [
        "computer_architecture",
        "computer_graphics",
    ],
    "4": [
        "theory_of_computation",
        "computer_networks",
        "operating_systems",
        "database_management_system",
        "artificial_intelligence",
    ],
    "5": [
        "system_analysis_and_design",
        "cryptography",
        "web_technology",
        "multimedia_computing",
    ],
    "6": [
        "technical_writing",
        "software_engineering",
        "egovernance",
        "dotnet_centric_computing",
        "ecommerce",
    ],
    "7": [
        "advanced_java_programming",
        "data_warehousing_and_mining",
    ],
    "8": [
        "advanced_database",
        "cloud_computing",
    ],
}

# Pretty names for display
SUBJECT_NAMES = {
    "introduction_to_it":            "Introduction to IT (CSC109)",
    "c_programming":                 "C Programming (CSC110)",
    "digital_logic":                 "Digital Logic (CSC111)",
    "physics":                       "Physics (PHY113)",
    "object_oriented_programming_cpp":"OOP in C++ (CSC161)",
    "microprocessor":                "Microprocessor (CSC162)",
    "computer_architecture":         "Computer Architecture (CSC208)",
    "computer_graphics":             "Computer Graphics (CSC209)",
    "theory_of_computation":         "Theory of Computation (CSC257)",
    "computer_networks":             "Computer Networks (CSC258)",
    "operating_systems":             "Operating Systems (CSC259)",
    "database_management_system":    "DBMS (CSC260)",
    "artificial_intelligence":       "Artificial Intelligence (CSC261)",
    "system_analysis_and_design":    "System Analysis & Design (CSC315)",
    "cryptography":                  "Cryptography (CSC316)",
    "web_technology":                "Web Technology (CSC318)",
    "multimedia_computing":          "Multimedia Computing",
    "software_engineering":          "Software Engineering (CSC364)",
    "egovernance":                   "E-Governance (CSC366)",
    "dotnet_centric_computing":      ".NET Centric Computing (CSC367)",
    "ecommerce":                     "E-Commerce (CSC368)",
    "advanced_java_programming":     "Advanced Java (CSC409)",
    "data_warehousing_and_mining":   "Data Warehousing & Mining (CSC410)",
    "advanced_database":             "Advanced Database (CSC461)",
    "cloud_computing":               "Cloud Computing (CSC467)",
}


# ================================================================
#  EPUB EXTRACTOR
# ================================================================
def extract_epub(path: Path, sem: str, subject: str) -> list[dict]:
    try:
        book     = epub.read_epub(str(path))
        chapters = []
        num      = 1

        for item in book.get_items():
            if item.get_type() != ebooklib.ITEM_DOCUMENT:
                continue
            soup = BeautifulSoup(item.get_content(), "html.parser")
            for tag in soup(["script", "style", "head"]):
                tag.decompose()
            text = soup.get_text(separator="\n")
            text = re.sub(r'\n{3,}', '\n\n', text).strip()
            if len(text) < 100:
                continue
            chapters.append({
                "text": text, "page": num,
                "file": path.name, "sem": sem, "subject": subject
            })
            num += 1

        print(f"       ✓ {path.name} → {len(chapters)} chapters")
        return chapters
    except Exception as e:
        print(f"       ❌ {path.name} failed: {e}")
        return []


# ================================================================
#  PDF EXTRACTOR (fallback)
# ================================================================
def extract_pdf(path: Path, sem: str, subject: str) -> list[dict]:
    try:
        doc   = fitz.open(str(path))
        pages = []
        for i in range(len(doc)):
            text = doc[i].get_text("text")
            if len(text.strip()) < 60:
                continue
            pages.append({
                "text": text, "page": i + 1,
                "file": path.name, "sem": sem, "subject": subject
            })
        doc.close()
        print(f"       ✓ {path.name} → {len(pages)} pages")
        return pages
    except Exception as e:
        print(f"       ❌ {path.name} failed: {e}")
        return []


# ================================================================
#  CLEAN + CHUNK
# ================================================================
def clean(text: str) -> str:
    text = re.sub(r'\n{3,}',    '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ',    text)
    text = re.sub(r'^\s*\d+\s*$', '',   text, flags=re.MULTILINE)
    return text.strip()


def chunk(pages: list[dict]) -> list[dict]:
    chunks = []
    idx    = 0
    for p in pages:
        text  = clean(p["text"])
        start = 0
        while start < len(text):
            piece = text[start: start + CHUNK_SIZE]
            if len(piece.strip()) > 80:
                chunks.append({
                    "id":      f"s{p['sem']}_{p['subject']}_{p['file']}_p{p['page']}_c{idx}",
                    "text":    piece.strip(),
                    "file":    p["file"],
                    "page":    p["page"],
                    "sem":     p["sem"],
                    "subject": p["subject"],
                })
                idx += 1
            start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


# ================================================================
#  STORE IN CHROMADB
# ================================================================
def store(chunks: list[dict], embedder):
    if not chunks:
        print("       ⚠  No chunks to store")
        return

    from collections import defaultdict
    by_col = defaultdict(list)
    for c in chunks:
        by_col[f"sem{c['sem']}_{c['subject']}"].append(c)

    client = get_chroma_client()

    for col_name, col_chunks in by_col.items():
        col      = client.get_or_create_collection(col_name, metadata={"hnsw:space": "cosine"})
        existing = set(col.get()["ids"])
        new      = [c for c in col_chunks if c["id"] not in existing]

        if not new:
            print(f"       ↩  {col_name}: already indexed")
            continue

        texts = [c["text"] for c in new]
        ids   = [c["id"]   for c in new]
        metas = [{"file": c["file"], "page": c["page"],
                  "sem": c["sem"], "subject": c["subject"]} for c in new]

        for i in tqdm(range(0, len(texts), 64), desc=f"       {col_name}"):
            embs = embedder.encode(texts[i:i+64], show_progress_bar=False).tolist()
            col.upsert(embeddings=embs, documents=texts[i:i+64],
                       ids=ids[i:i+64], metadatas=metas[i:i+64])

        print(f"       ✅ {col_name}: {len(new)} chunks stored (total: {col.count()})")


# ================================================================
#  MAIN
# ================================================================
def run(sem_arg: str, subject_arg: str = None):
    print(f"\nLoading: {EMBED_MODEL}  (downloads once ~400MB)")
    embedder = SentenceTransformer(EMBED_MODEL)

    sems = list(SEMESTER_SUBJECTS.keys()) if sem_arg == "all" else [sem_arg]

    for sem in sems:
        subjects = SEMESTER_SUBJECTS.get(sem, [])
        if not subjects:
            print(f"⚠  Semester '{sem}' not found. Valid: 1-8 or 'all'")
            continue

        print(f"\n{'='*55}\n  SEMESTER {sem}\n{'='*55}")

        for subject in subjects:
            if subject_arg and subject != subject_arg:
                continue

            folder = BASE_DIR / "Data" / f"Sem{sem}" / subject
            if not folder.exists():
                print(f"\n  ⚠  {SUBJECT_NAMES.get(subject, subject)}")
                print(f"     Folder missing → create: {folder}")
                continue

            files = list(folder.glob("*.epub")) + list(folder.glob("*.pdf"))
            if not files:
                print(f"\n  ⚠  {SUBJECT_NAMES.get(subject, subject)}")
                print(f"     No epub/pdf found in {folder}")
                continue

            print(f"\n  📚 {SUBJECT_NAMES.get(subject, subject)}  ({len(files)} file(s))")
            all_pages = []
            for f in files:
                if f.suffix.lower() == ".epub":
                    all_pages.extend(extract_epub(f, sem, subject))
                else:
                    all_pages.extend(extract_pdf(f, sem, subject))

            chunks = chunk(all_pages)
            print(f"       → {len(chunks)} total chunks")
            store(chunks, embedder)

    print("\n✅ Done! Run:  uv run streamlit run app.py")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sem",     required=True)
    p.add_argument("--subject", default=None)
    args = p.parse_args()
    run(args.sem, args.subject)
