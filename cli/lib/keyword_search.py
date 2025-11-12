from typing import List
from .inverted_index import InvertedIndex
from .schemas import MovieModel
from .search_utils import DEFAULT_SEARCH_LIMIT, preprocess_text


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


def tf_command(doc_id: int, term: str) -> int:
    token = preprocess_text(term)
    if len(token) > 1:
        raise ValueError(f"Only one word is searchable for count")

    idx = InvertedIndex()
    idx.load()

    return idx.get_tf(doc_id, term)


def idx_command(term: str) -> float:
    idx = InvertedIndex()
    idx.load()

    return idx.get_idf(term)
