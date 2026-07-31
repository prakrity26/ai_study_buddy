# ================================================================
#  ingest.py  —  TU BSc CSIT AI Study Buddy
#  Reads EPUB / PDF files → chunks → stores in ChromaDB
#
#  USAGE:
#    Single subject:  uv run python src/ingest.py --sem 4 --subject operating_systems
#    Full semester:   uv run python src/ingest.py --sem 4
#    Everything:      uv run python src/ingest.py --sem all
#
#  SUPPORTED DATA STRUCTURES (both are scanned automatically):
#    1. Data/Semester IV/CSC264 - Operating Systems/*.epub   <- existing books
#    2. Data/Sem4/operating_systems/*.epub                   <- UI uploads
# ================================================================

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
    from src.chroma_client import get_chroma_client
except ModuleNotFoundError:
    from chroma_client import get_chroma_client

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env", override=False)

EMBED_MODEL   = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE",    600))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP",  80))

# ── Roman numeral -> integer semester mapping ────────────────────
_ROMAN = {
    "I": "1", "II": "2", "III": "3", "IV": "4",
    "V": "5", "VI": "6", "VII": "7", "VIII": "8",
}

# ── Slug overrides for non-obvious normalizations ────────────────
_SLUG_OVERRIDES = {
    "design and analysis of algorithms":          "design_and_analysis_of_algorithms",
    "system analysis and design":                 "system_analysis_and_design",
    "simulation and modeling":                    "simulation_and_modeling",
    "net centric computing":                      "dotnet_centric_computing",
    "e-commerce":                                 "ecommerce",
    "data warehousing and data mining":           "data_warehousing_and_mining",
    "network and system administration":          "network_and_system_administration",
    "geographical information system":            "geographical_information_system",
    "introduction to cloud computing":            "cloud_computing",
    "advanced networking with ipv6":              "advanced_networking_ipv6",
    "distributed and object oriented database":   "distributed_and_object_oriented_database",
    "compiler design and construction":           "compiler_design",
    "applied logic":                              "applied_logic",
    "database administration":                    "database_administration",
    "principles of management":                   "principles_of_management",
    "software project management":                "software_project_management",
    "network security":                           "network_security",
    "game technology":                            "game_technology",
    "distributed networking":                     "distributed_networking",
    "mobile application development":             "mobile_application_development",
    "real time systems":                          "real_time_systems",
    "embedded systems programming":               "embedded_systems_programming",
}

# ── Pretty display names keyed by slug ──────────────────────────
SUBJECT_NAMES = {
    "introduction_to_it":                       "Introduction to IT (CSC109)",
    "c_programming":                            "C Programming (CSC110)",
    "digital_logic":                            "Digital Logic (CSC111)",
    "physics":                                  "Physics (PHY113)",
    "discrete_structures":                      "Discrete Structures (CSC160)",
    "object_oriented_programming_cpp":          "OOP in C++ (CSC161)",
    "microprocessor":                           "Microprocessor (CSC162)",
    "computer_architecture":                    "Computer Architecture (CSC208)",
    "computer_graphics":                        "Computer Graphics (CSC209)",
    "theory_of_computation":                    "Theory of Computation (CSC262)",
    "computer_networks":                        "Computer Networks (CSC263)",
    "operating_systems":                        "Operating Systems (CSC264)",
    "database_management_system":               "DBMS (CSC265)",
    "artificial_intelligence":                  "Artificial Intelligence (CSC266)",
    "design_and_analysis_of_algorithms":        "Design & Analysis of Algorithms (CSC325)",
    "system_analysis_and_design":               "System Analysis & Design (CSC326)",
    "cryptography":                             "Cryptography (CSC327)",
    "simulation_and_modeling":                  "Simulation and Modeling (CSC328)",
    "web_technology":                           "Web Technology (CSC329)",
    "multimedia_computing":                     "Multimedia Computing",
    "software_engineering":                     "Software Engineering (CSC375)",
    "compiler_design":                          "Compiler Design (CSC376)",
    "dotnet_centric_computing":                 ".NET Centric Computing (CSC378)",
    "applied_logic":                            "Applied Logic (CSC380)",
    "ecommerce":                                "E-Commerce (CSC381)",
    "egovernance":                              "E-Governance",
    "technical_writing":                        "Technical Writing",
    "advanced_java_programming":                "Advanced Java (CSC419)",
    "data_warehousing_and_mining":              "Data Warehousing & Mining (CSC420)",
    "database_administration":                  "Database Administration (CSC424)",
    "software_project_management":              "Software Project Management (CSC425)",
    "network_security":                         "Network Security (CSC426)",
    "principles_of_management":                 "Principles of Management (MGT421)",
    "advanced_database":                        "Advanced Database (CSC475)",
    "advanced_networking_ipv6":                 "Advanced Networking with IPv6 (CSC477)",
    "distributed_networking":                   "Distributed Networking (CSC478)",
    "game_technology":                          "Game Technology (CSC479)",
    "distributed_and_object_oriented_database": "Distributed & OO Database (CSC480)",
    "cloud_computing":                          "Cloud Computing (CSC481)",
    "geographical_information_system":          "Geographical Information System (CSC482)",
    "mobile_application_development":           "Mobile Application Development (CSC484)",
    "real_time_systems":                        "Real Time Systems (CSC485)",
    "network_and_system_administration":        "Network & System Administration (CSC486)",
    "embedded_systems_programming":             "Embedded Systems Programming (CSC487)",
}


def _slug(name: str) -> str:
    normalized = name.strip().lower()
    return _SLUG_OVERRIDES.get(normalized, re.sub(r"[^a-z0-9]+", "_", normalized).strip("_"))


# ================================================================
#  FILE DISCOVERY — scans both folder structures
# ================================================================
def discover_files(sem_arg: str, subject_arg: str = None) -> list[dict]:
    """Return [{sem, subject, label, files}] from all data folder structures."""
    results   = []
    seen_keys = set()  # avoid double-indexing same (sem, subject)

    # Structure 1: Data/Semester IV/CSC264 - Operating Systems/
    for roman, num in _ROMAN.items():
        if sem_arg != "all" and num != sem_arg:
            continue
        sem_dir = BASE_DIR / "Data" / f"Semester {roman}"
        if not sem_dir.exists():
            continue
        for folder in sorted(p for p in sem_dir.iterdir() if p.is_dir()):
            files = sorted(
                f for f in folder.iterdir()
                if f.is_file() and f.suffix.lower() in {".epub", ".pdf"}
            )
            if not files:
                continue
            match = re.match(r"^[A-Z]+\d+\s*-\s*(.+)$", folder.name)
            name  = match.group(1).strip() if match else folder.name
            slug  = _slug(name)
            if subject_arg and slug != subject_arg:
                continue
            key = (num, slug)
            if key not in seen_keys:
                seen_keys.add(key)
                results.append({
                    "sem":     num,
                    "subject": slug,
                    "label":   SUBJECT_NAMES.get(slug, name),
                    "files":   files,
                })

    # Structure 2: Data/Sem4/operating_systems/ (UI uploads)
    for num in (sorted(_ROMAN.values()) if sem_arg == "all" else [sem_arg]):
        flat_dir = BASE_DIR / "Data" / f"Sem{num}"
        if not flat_dir.exists():
            continue
        for folder in sorted(p for p in flat_dir.iterdir() if p.is_dir()):
            slug = folder.name
            if subject_arg and slug != subject_arg:
                continue
            files = sorted(
                f for f in folder.iterdir()
                if f.is_file() and f.suffix.lower() in {".epub", ".pdf"}
            )
            if not files:
                continue
            key = (num, slug)
            if key not in seen_keys:
                seen_keys.add(key)
                results.append({
                    "sem":     num,
                    "subject": slug,
                    "label":   SUBJECT_NAMES.get(slug, slug.replace("_", " ").title()),
                    "files":   files,
                })

    return results


# ================================================================
#  EPUB EXTRACTOR
# ================================================================
def extract_epub(path: Path, sem: str, subject: str) -> list[dict]:
    try:
        book, chapters, num = epub.read_epub(str(path)), [], 1
        for item in book.get_items():
            if item.get_type() != ebooklib.ITEM_DOCUMENT:
                continue
            soup = BeautifulSoup(item.get_content(), "html.parser")
            for tag in soup(["script", "style", "head"]):
                tag.decompose()
            blocks = soup.find_all(["h1","h2","h3","h4","h5","h6","p","li","blockquote"])
            text   = "\n\n".join(b.get_text(" ", strip=True) for b in blocks).strip()
            if not text:
                text = soup.get_text(separator="\n\n").strip()
            if len(text) < 100:
                continue
            chapters.append({
                "text": text, "page": num,
                "file": path.name, "sem": sem, "subject": subject,
            })
            num += 1
        print(f"       + {path.name} -> {len(chapters)} sections")
        return chapters
    except Exception as exc:
        print(f"       x {path.name} failed: {exc}")
        return []


# ================================================================
#  PDF EXTRACTOR
# ================================================================
def extract_pdf(path: Path, sem: str, subject: str) -> list[dict]:
    try:
        doc, pages = fitz.open(str(path)), []
        for i in range(len(doc)):
            text = doc[i].get_text("text").strip()
            if len(text) < 60:
                continue
            pages.append({
                "text": text, "page": i + 1,
                "file": path.name, "sem": sem, "subject": subject,
            })
        doc.close()
        print(f"       + {path.name} -> {len(pages)} pages")
        return pages
    except Exception as exc:
        print(f"       x {path.name} failed: {exc}")
        return []


# ================================================================
#  CLEAN + CHUNK
# ================================================================
def _clean(text: str) -> str:
    text = re.sub(r"\n{3,}",    "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ",    text)
    return re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE).strip()


def chunk(pages: list[dict]) -> list[dict]:
    chunks = []
    idx    = 0
    for p in pages:
        text  = _clean(p["text"])
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
def store(chunks: list[dict], embedder) -> None:
    if not chunks:
        print("       !  No chunks to store")
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
            print(f"       = {col_name}: already indexed")
            continue

        texts = [c["text"] for c in new]
        ids   = [c["id"]   for c in new]
        metas = [{"file": c["file"], "page": c["page"],
                  "sem": c["sem"], "subject": c["subject"]} for c in new]

        for i in tqdm(range(0, len(texts), 64), desc=f"       {col_name}"):
            embs = embedder.encode(texts[i:i+64], show_progress_bar=False).tolist()
            col.upsert(embeddings=embs, documents=texts[i:i+64],
                       ids=ids[i:i+64], metadatas=metas[i:i+64])

        print(f"       OK {col_name}: {len(new)} new chunks (total: {col.count()})")


# ================================================================
#  MAIN
# ================================================================
def run(sem_arg: str, subject_arg: str = None) -> None:
    print(f"\nLoading embedding model: {EMBED_MODEL}")
    embedder = SentenceTransformer(EMBED_MODEL)

    subjects = discover_files(sem_arg, subject_arg)
    if not subjects:
        print(f"\n  No EPUB/PDF files found for sem={sem_arg!r} subject={subject_arg!r}")
        print("  Make sure books are in Data/Semester IV/... or Data/Sem4/.../")
        return

    for entry in subjects:
        print(f"\n  {entry['label']}  ({len(entry['files'])} file(s))")
        all_pages = []
        for f in entry["files"]:
            extractor = extract_epub if f.suffix.lower() == ".epub" else extract_pdf
            all_pages.extend(extractor(f, entry["sem"], entry["subject"]))
        chunks_list = chunk(all_pages)
        print(f"       -> {len(chunks_list)} total chunks")
        store(chunks_list, embedder)

    print("\nDone! Run:  uv run streamlit run app.py")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sem",     required=True, help="Semester number (1-8) or 'all'")
    p.add_argument("--subject", default=None,  help="Subject slug (optional)")
    args = p.parse_args()
    run(args.sem, args.subject)
