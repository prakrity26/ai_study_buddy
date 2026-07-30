# AI Study Buddy

AI Study Buddy is a retrieval-augmented generation (RAG) application for
Tribhuvan University BSc CSIT students. It searches uploaded EPUB and PDF
course material, reranks the most relevant passages, and uses local Hugging
Face models to produce answers grounded in those sources.

The Streamlit app runs locally with `uv`. ChromaDB runs in Docker and stores
vectors in a Docker volume.

## Features

- Streamlit chat interface
- Search across all indexed course material or filter by semester and subject
- Hybrid semantic and BM25 keyword retrieval
- FlashRank passage reranking
- Answers with textbook and chapter/page references
- EPUB and PDF ingestion
- ChromaDB vector storage running in Docker

## Requirements

- Docker Desktop or Docker Engine with Docker Compose
- uv
- Enough local memory to run Qwen 3.5 4B (a GPU or Apple Silicon is recommended)

## Quick Start

Create a `.env` file in the project root:

```dotenv
CHROMA_HOST=localhost
CHROMA_PORT=8000
```

Start ChromaDB:

```bash
docker compose up -d
```

Install the Python environment:

```bash
uv sync
```

Start the Streamlit app:

```bash
uv run streamlit run app.py
```

Open:

```text
http://localhost:8501
```

The first app run downloads the embedding model, reranker, and local Qwen 3.5
4B model. This can take several gigabytes; the model stays in the local Hugging
Face cache for later runs.

## Add Study Material

You can select a semester and upload EPUB or PDF files from the application
sidebar. To prepare files manually, use this directory structure:

```text
Data/
└── Sem6/
    └── software_engineering/
        ├── textbook.epub
        └── notes.pdf
```

Folder names must use the subject slugs defined in `src/ingest.py`, such as
`software_engineering`, `operating_systems`, or
`database_management_system`.

Index one subject:

```bash
uv run python src/ingest.py --sem 6 --subject software_engineering
```

Index every available subject in a semester:

```bash
uv run python src/ingest.py --sem 6
```

Index all available semesters:

```bash
uv run python src/ingest.py --sem all
```

Indexed vectors are stored in the `chroma_data` Docker volume.

## Configuration

All settings are optional.

| Variable | Default | Purpose |
| --- | --- | --- |
| `LLM_MODEL` | `Qwen/Qwen3.5-4B` | Local Hugging Face model used to generate answers |
| `EMBEDDING_MODEL` | `google/embeddinggemma-300m` | Local Sentence Transformer model used to create and search vectors |
| `LLM_MAX_NEW_TOKENS` | `900` | Maximum length of each answer |
| `CHROMA_HOST` | unset | Chroma server host. Use `localhost` for Docker Chroma |
| `CHROMA_PORT` | `8000` | Chroma server port |
| `CHROMA_PATH` | `VectorStore/chroma_db` | Local fallback path when `CHROMA_HOST` is unset |
| `CHROMA_CONNECT_RETRIES` | `30` | Chroma HTTP connection retries at startup |
| `CHROMA_CONNECT_INTERVAL` | `1` | Seconds between Chroma connection retries |
| `TOP_K` | `10` | Retrieval candidates per search method |
| `RERANK_TOP` | `4` | Passages retained after reranking |
| `CHUNK_SIZE` | `600` | Characters per indexed chunk |
| `CHUNK_OVERLAP` | `80` | Overlap between adjacent chunks |

Example `.env`:

```dotenv
CHROMA_HOST=localhost
CHROMA_PORT=8000
LLM_MODEL=Qwen/Qwen3.5-4B
EMBEDDING_MODEL=google/embeddinggemma-300m
TOP_K=10
RERANK_TOP=4
```

### Re-index after changing the embedding model

Vectors created by one embedding model cannot be searched reliably by another.
If this project was previously indexed with `BAAI/bge-base-en-v1.5`, remove the
old Chroma volume before ingesting again:

```bash
docker compose down -v
docker compose up -d
uv run python src/ingest.py --sem all
```

## Useful Docker Commands

Start Chroma:

```bash
docker compose up -d
```

Stop Chroma:

```bash
docker compose down
```

Stop Chroma and remove indexed vectors:

```bash
docker compose down -v
```

View Chroma logs:

```bash
docker compose logs -f chroma
```

## Evaluation

The evaluation script checks generated answers against expected keywords:

```bash
uv run python Evaluation/evaluate.py
```

Evaluation requires the corresponding subjects to be indexed and enough local
memory to load the Hugging Face models.

## Project Structure

```text
.
├── app.py                  # Streamlit user interface
├── docker-compose.yml      # ChromaDB service
├── src/
│   ├── chroma_client.py    # ChromaDB connection helper
│   ├── ingest.py           # EPUB/PDF extraction and indexing
│   └── rag_engine.py       # Retrieval, reranking, and answer generation
├── Evaluation/
│   └── evaluate.py         # Keyword-based answer evaluation
├── Data/                   # Local course material
├── pyproject.toml          # Project metadata and direct dependencies
└── uv.lock                 # Reproducible dependency lockfile
```

## Troubleshooting

### Docker Cannot Connect

Start Docker Desktop, then run:

```bash
docker compose up -d
```

### Local model runs out of memory

Qwen 3.5 4B is loaded locally and needs substantially more memory than the
previous API-based setup. Close other memory-heavy applications, use a GPU or
Apple Silicon machine with sufficient unified memory, or set `LLM_MODEL` to a
smaller compatible local model.

### No Study Material Found

Check that the selected semester and subject have been indexed. Run the relevant
ingestion command again and confirm that it reports stored chunks.

### Chroma Telemetry Logs

If you see `Failed to send telemetry event ... capture() takes 1 positional
argument but 3 were given`, refresh the uv environment:

```bash
uv sync
```

The project pins PostHog below version 3 because ChromaDB 0.5.3 uses the older
PostHog Python API.
