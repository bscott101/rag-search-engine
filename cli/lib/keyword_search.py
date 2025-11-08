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

    query_tokens = preprocess_text(query)
    seen, results = set(), []
    for token in query_tokens:
        doc_ids = idx.get_documents(token)
        for id in doc_ids:
            if id in seen:
                continue
            doc = idx.get_document_object(id)
            results.append(doc)
            if len(results) >= limit:
                return results

    return results


def build_command():
    idx = InvertedIndex()
    idx.build()
    idx.save()
