# AI Study Buddy

AI Study Buddy is a retrieval-augmented generation (RAG) application for
Tribhuvan University BSc CSIT students. It searches uploaded EPUB and PDF
course material, reranks the most relevant passages, and uses Groq to produce
answers grounded in those sources.

## Features

- Search across all indexed course material or filter by semester and subject
- Hybrid semantic and BM25 keyword retrieval
- FlashRank passage reranking
- Answers with textbook and chapter/page references
- EPUB and PDF ingestion
- Streamlit interface with in-app uploads
- Local persistent vector storage using ChromaDB

## Requirements

- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- A [Groq API key](https://console.groq.com/keys)

The project uses Python 3.11 by default. `uv` will install a compatible Python
version automatically when needed.

## Quick start

Clone the repository and enter the project directory, then install the locked
dependencies:

```bash
uv sync
```

Create a `.env` file in the project root:

```dotenv
GROQ_API_KEY=your_groq_api_key
```

Start the application:

```bash
uv run streamlit run app.py
```

Open the local URL shown by Streamlit, usually
`http://localhost:8501`.

> [!NOTE]
> The first run downloads the embedding and reranking models. Later runs reuse
> the local model cache.

## Add study material

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

Indexed content is stored under `VectorStore/` and is excluded from Git.

## Configuration

All settings are optional except `GROQ_API_KEY`.

| Variable | Default | Purpose |
| --- | --- | --- |
| `GROQ_API_KEY` | — | Authenticates requests to Groq |
| `LLM_MODEL` | `llama-3.1-8b-instant` | Groq model used to generate answers |
| `EMBEDDING_MODEL` | `BAAI/bge-base-en-v1.5` | Sentence Transformer embedding model |
| `CHROMA_PATH` | `VectorStore/chroma_db` | ChromaDB storage location |
| `TOP_K` | `10` | Retrieval candidates per search method |
| `RERANK_TOP` | `4` | Passages retained after reranking |
| `CHUNK_SIZE` | `600` | Characters per indexed chunk |
| `CHUNK_OVERLAP` | `80` | Overlap between adjacent chunks |

Example:

```dotenv
GROQ_API_KEY=your_groq_api_key
LLM_MODEL=llama-3.1-8b-instant
TOP_K=10
RERANK_TOP=4
```

## Evaluation

The evaluation script checks generated answers against expected keywords:

```bash
uv run python Evaluation/evaluate.py
```

Evaluation requires the corresponding subjects to be indexed and a working
Groq API key.

## Project structure

```text
.
├── app.py                  # Streamlit user interface
├── src/
│   ├── ingest.py           # EPUB/PDF extraction and indexing
│   └── rag_engine.py       # Retrieval, reranking, and answer generation
├── Evaluation/
│   └── evaluate.py         # Keyword-based answer evaluation
├── Data/                   # Local course material
├── VectorStore/            # Generated ChromaDB and reranker cache
├── pyproject.toml          # Project metadata and direct dependencies
└── uv.lock                 # Reproducible dependency lockfile
```

## Troubleshooting

### `uv: command not found`

Install `uv` using the
[official installation instructions](https://docs.astral.sh/uv/getting-started/installation/),
then reopen the terminal.

### Missing Groq API key

Confirm that `.env` is in the project root and contains:

```dotenv
GROQ_API_KEY=your_groq_api_key
```

Restart Streamlit after changing the file.

### No study material found

Check that the selected semester and subject have been indexed. Run the
relevant ingestion command again and confirm that it reports stored chunks.

### Recreate the environment

Ask `uv` to restore the environment from the lockfile:

```bash
uv sync --locked
```
