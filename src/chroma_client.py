import os
import time
from pathlib import Path

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env", override=False)

CHROMA_PATH = os.getenv("CHROMA_PATH", str(BASE_DIR / "VectorStore" / "chroma_db"))


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def get_chroma_client():
    host = os.getenv("CHROMA_HOST")
    settings = Settings(anonymized_telemetry=False)

    if host:
        retries = int(os.getenv("CHROMA_CONNECT_RETRIES", "30"))
        delay = float(os.getenv("CHROMA_CONNECT_INTERVAL", "1"))
        last_error = None

        for attempt in range(retries):
            try:
                return chromadb.HttpClient(
                    host=host,
                    port=int(os.getenv("CHROMA_PORT", "8000")),
                    ssl=_env_bool("CHROMA_SSL", False),
                    settings=settings,
                )
            except Exception as exc:
                last_error = exc
                if attempt < retries - 1:
                    time.sleep(delay)

        raise last_error

    return chromadb.PersistentClient(path=CHROMA_PATH, settings=settings)
