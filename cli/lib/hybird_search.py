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
        safe_limit = limit * 500
        bm25_search = self._bm25_search(query, safe_limit)
        chunked_search = self._semantic_chunk_search(query, safe_limit)

        # sort the values so they match up
        bm25_search_sorted = sorted(bm25_search, key=lambda x: x["id"])
        chunked_search_sorted = sorted(chunked_search, key=lambda x: x["id"])

        bm25_scores = [x["score"] for x in bm25_search_sorted]
        bm25_norm_scores = normalize_scores(bm25_scores)

        chunked_scores = [x["score"] for x in chunked_search_sorted]
        chunked_norm_scores = normalize_scores(chunked_scores)

        scores = []
        doc_scores = {}
        for index in range(0, len(bm25_norm_scores)):
            if bm25_search_sorted[index]["id"] != chunked_search_sorted[index]["id"]:
                raise ("Something fucked up")

            doc = bm25_search_sorted[index]
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

    def rrf_search(self, query, k, limit=10) -> List[dict]:
        safe_limit = limit * 500
        bm25_search = self._bm25_search(query, safe_limit)
        bm25_sorted = sorted(bm25_search, key=lambda x: x["score"], reverse=True)

        chunked_search = self._semantic_chunk_search(query, safe_limit)
        chunked_sorted = sorted(chunked_search, key=lambda x: x["score"], reverse=True)

        result = {}
        for rank, bm25_res in enumerate(bm25_sorted, start=1):
            doc_id = bm25_res["id"]
            if doc_id not in result:
                result[doc_id] = {
                    "title": bm25_res["title"],
                    "document": bm25_res["document"],
                    "bm25_rank": rank,
                    "semantic_rank": 0,
                }
            if result[doc_id]["bm25_rank"] > rank or result[doc_id]["bm25_rank"] == 0:
                result[doc_id]["bm25_rank"] = rank

        for rank, sem_res in enumerate(chunked_sorted, start=1):
            doc_id = sem_res["id"]
            if doc_id not in result:
                result[doc_id] = {
                    "title": sem_res["title"],
                    "document": sem_res["document"],
                    "bm25_rank": 0,
                    "semantic_rank": rank,
                }
            if (
                result[doc_id]["semantic_rank"] > rank
                or result[doc_id]["semantic_rank"] == 0
            ):
                result[doc_id]["semantic_rank"] = rank

        scores = []
        for doc_id, res in result.items():
            bm25_rrf = rrf_score(res["bm25_rank"], k=k)
            sem_rrf = rrf_score(res["semantic_rank"], k=k)
            score = bm25_rrf + sem_rrf
            res["rrf_score"] = score
            scores.append((doc_id, score))

        scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)[:limit]
        return [result[id] for id, _ in scores_sorted]

    def _semantic_chunk_search(
        self, query: str, limit: int = DEFAULT_SEARCH_LIMIT
    ) -> dict:
        return self.semantic_search.search_chunks(query, limit)


def rrf_score(rank, k=60) -> float:
    return 1 / (k + rank)


def semantic_chunk_search(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> dict:
    model = HybirdSearch(load_movies())
    return model._semantic_chunk_search(query, limit)


def normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)

    if max_score == min_score:
        return [1.0] * len(scores)

    normalized_scores = []
    for s in scores:
        normalized_scores.append((s - min_score) / (max_score - min_score))

    return normalized_scores


def hybrid_score(bm25_score: float, semantic_score: float, alpha: float = 0.5):
    return alpha * bm25_score + (1 - alpha) * semantic_score


def weighted_search(query: str, alpha: float = 0.5, limit: int = 5) -> List[dict]:
    model = HybirdSearch(load_movies())
    return model.weighted_search(query, alpha, limit)


def rrf_search_command(query: str, k: int = 60, limit: int = 5) -> List[dict]:
    model = HybirdSearch(load_movies())
    return model.rrf_search(query, k, limit)
