# AI Study Buddy

AI Study Buddy is a retrieval-augmented generation (RAG) application for
Tribhuvan University BSc CSIT students. It searches uploaded EPUB and PDF
course material, reranks the most relevant passages, and uses Groq to produce
answers grounded in those sources.

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
- A Groq API key

## Quick Start

Create a `.env` file in the project root:

```dotenv
GROQ_API_KEY=your_groq_api_key
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

The first app run downloads the embedding and reranking models.

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

All settings are optional except `GROQ_API_KEY`.

| Variable | Default | Purpose |
| --- | --- | --- |
| `GROQ_API_KEY` | required | Authenticates requests to Groq |
| `LLM_MODEL` | `llama-3.1-8b-instant` | Groq model used to generate answers |
| `EMBEDDING_MODEL` | `BAAI/bge-base-en-v1.5` | Sentence Transformer embedding model |
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
GROQ_API_KEY=your_groq_api_key
CHROMA_HOST=localhost
CHROMA_PORT=8000
LLM_MODEL=llama-3.1-8b-instant
TOP_K=10
RERANK_TOP=4
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

Evaluation requires the corresponding subjects to be indexed and a working Groq
API key.

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

### Missing Groq API Key

Confirm that `.env` is in the project root and contains:

```dotenv
GROQ_API_KEY=your_groq_api_key
```

Restart the app after changing the file:

```bash
uv run streamlit run app.py
```

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
