import os

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from .schemas import Movie
from .keyword_search import DEFAULT_SEARCH_LIMIT, load_movies

from typing import List


class HybirdSearch:
    def __init__(self, documents: List[Movie]):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float = 0.5, limit: int = 5):
        bm25_search = self._bm25_search(query, limit * 500)
        chunked_search = self._semantic_chunk_search(query, limit * 500)

        bm25_scores = [x["score"] for x in bm25_search]
        bm25_norm_scores = normalize_scores(bm25_scores)
        chunked_scores = [x["score"] for x in chunked_search["result"]]
        chunked_norm_scores = normalize_scores(chunked_scores)

        scores = []
        doc_scores = {}
        for index in range(0, len(bm25_norm_scores)):
            doc = bm25_search[index]
            bm25_score = bm25_norm_scores[index]
            semantic_score = chunked_norm_scores[index]
            hybird = hybrid_score(bm25_score, semantic_score, alpha)

            scores.append((doc["id"], hybird))
            doc_scores[doc["id"]] = {
                "bm25": bm25_score,
                "semantic": semantic_score,
                "hybrid": hybird,
                "title": doc["title"],
                "document": doc["document"],
            }
        scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)[:limit]
        return [doc_scores[id] for id, _ in scores_sorted]
        # raise NotImplementedError("Weighted hybrid search is not implemented yet.")

    def rrf_search(self, query, k, limit=10):
        raise NotImplementedError("RRF hybrid search is not implemented yet.")

    def _semantic_chunk_search(
        self, query: str, limit: int = DEFAULT_SEARCH_LIMIT
    ) -> dict:
        return self.semantic_search.search_chunks(query, limit)


def semantic_chunk_search(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> dict:
    model = HybirdSearch(load_movies())
    return model._semantic_chunk_search(query, limit)


def normalize_scores(scores: list[float]) -> list[float]:
    if len(scores) == 0:
        return []

    result = []
    min_score = min(scores)
    max_score = max(scores)

    if min_score == max_score:
        return [1.0] * len(scores)

    for score in scores:
        result.append((score - min_score) / (max_score - min_score))

    return result


def hybrid_score(bm25_score, semantic_score, alpha=0.5):
    return alpha * bm25_score + (1 - alpha) * semantic_score


def weighted_search(query: str, alpha: float = 0.5, limit: int = 5) -> List[dict]:
    model = HybirdSearch(load_movies())
    return model.weighted_search(query, alpha, limit)
