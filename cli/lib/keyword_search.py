from .search_utils import (
    load_movies,
    DEFAULT_SEARCH_LIMIT,
    preprocess_text,
)
from .inverted_index import InvertedIndex
from typing import List, Dict


def has_matching_token(query_tokens: List[str], title_tokens: List[str]) -> bool:
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> List[Dict]:
    movies = load_movies()
    results = []
    for movie in movies:
        query_tokens = preprocess_text(query)
        title_tokens = preprocess_text(movie.title)
        if has_matching_token(query_tokens, title_tokens):
            results.append(movie)
            if len(results) >= limit:
                break

    return results


def build_command():
    idx = InvertedIndex()
    idx.build()
    idx.save()
    docs = idx.get_documents("merida")
    print(f"First document for token 'merida' = {docs[0]}")
