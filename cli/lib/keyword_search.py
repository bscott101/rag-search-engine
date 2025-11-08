from .search_utils import (
    load_movies,
    DEFAULT_SEARCH_LIMIT,
    preprocess_text,
)
from .schemas import MovieModel
from .inverted_index import InvertedIndex
from typing import List


def has_matching_token(query_tokens: List[str], title_tokens: List[str]) -> bool:
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> List[MovieModel]:
    idx = InvertedIndex()
    idx.load()

    movie_ids = idx.get_documents(query)
    results = []
    for id in movie_ids[:limit]:
        movie = idx.get_document_object(id)
        results.append(movie)

    return results


def build_command():
    idx = InvertedIndex()
    idx.build()
    idx.save()
