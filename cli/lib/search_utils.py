import json
import os
from typing import Any, List
from .schemas import Movies, Movie

DEFAULT_SEARCH_LIMIT = 5
DOCUMENT_PREVIEW_LENGTH = 100
SCORE_PRECISION = 3

BM25_K1 = 1.5
BM25_B = 0.75

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")

CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")

DEFAULT_CHUNK_SIZE = 200
DEFAULT_CHUNK_OVERLAP = 1
DEFAULT_SEMANTIC_CHUNK_SIZE = 4
DEFAULT_SEMANTIC_LIMIT = 5


def load_movies() -> List[Movie]:
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
        data = Movies(**data)
    return data.movies


def load_stopwords() -> list[str]:
    with open(STOPWORDS_PATH, "r") as f:
        return f.read().splitlines()


def format_search_result(
    doc_id: int, title: str, document: str, score: float, **metadata: Any
) -> dict[str, Any]:

    return {
        "id": doc_id,
        "title": title,
        "document": document,
        "score": round(score, SCORE_PRECISION),
        "metadata": metadata if metadata else {},
    }
