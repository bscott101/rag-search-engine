import os
from typing import Optional

from .keyword_search import InvertedIndex
from .query_enhancement import enhance_query
from .semantic_search import ChunkedSemanticSearch
from .schemas import Movie, FormattedResults
from .keyword_search import DEFAULT_SEARCH_LIMIT, load_movies
from .search_utils import SEARCH_MULTIPLIER
from .reranking import rerank_command

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

    def weighted_search(
        self, query: str, alpha: float = 0.5, limit: int = 5
    ) -> List[FormattedResults]:
        safe_limit = limit * 500
        bm25_search = self._bm25_search(query, safe_limit)
        chunked_search = self._semantic_chunk_search(query, safe_limit)
        results: dict[str, FormattedResults] = {}

        bm25_norm_scores = normalise_search_results(bm25_search)
        for doc in bm25_norm_scores:
            if doc.doc_id not in results:
                results[doc.doc_id] = FormattedResults(
                    doc_id=doc.doc_id,
                    title=doc.title,
                    document=doc.document,
                    score=doc.score,
                    bm25_score=doc.norm_score,
                )
            if (
                results[doc.doc_id].bm25_score is None
                or doc.norm_score > results[doc.doc_id].bm25_score
            ):
                results[doc.doc_id].bm25_score = doc.norm_score

        chunked_norm_scores = normalise_search_results(chunked_search)
        for doc in chunked_norm_scores:
            if doc.doc_id not in results:
                results[doc.doc_id] = FormattedResults(
                    doc_id=doc.doc_id,
                    title=doc.title,
                    document=doc.document,
                    score=doc.score,
                    semantic_score=doc.norm_score,
                )
            if (
                results[doc.doc_id].semantic_score is None
                or doc.norm_score > results[doc.doc_id].semantic_score
            ):
                results[doc.doc_id].semantic_score = doc.norm_score

        hybird_scores: list[FormattedResults] = []
        for _, doc in results.items():
            score = hybrid_score(doc.bm25_score, doc.semantic_score, alpha)
            doc.score = score
            doc.hybird_score = score
            hybird_scores.append(doc)

        return sorted(hybird_scores, key=lambda x: x.hybird_score, reverse=True)

    def rrf_search(self, query, k, limit=10) -> List[dict]:
        safe_limit = limit * 500
        bm25_search = self._bm25_search(query, safe_limit)
        bm25_sorted = sorted(bm25_search, key=lambda x: x.score, reverse=True)

        chunked_search = self._semantic_chunk_search(query, safe_limit)
        chunked_sorted = sorted(chunked_search, key=lambda x: x.score, reverse=True)

        results: dict[str, FormattedResults] = {}
        for rank, doc in enumerate(bm25_sorted, start=1):
            doc_id = doc.doc_id
            if doc_id not in results:
                results[doc.doc_id] = FormattedResults(
                    doc_id=doc.doc_id,
                    title=doc.title,
                    document=doc.document,
                    score=doc.score,
                    bm25_rank=rank,
                    bm25_score=doc.score,
                )
            if results[doc.doc_id].bm25_rank < 0 or results[doc_id].bm25_rank > rank:
                results[doc_id].bm25_rank = rank

        for rank, doc in enumerate(chunked_sorted, start=1):
            doc_id = doc.doc_id
            if doc_id not in results:
                results[doc.doc_id] = FormattedResults(
                    doc_id=doc.doc_id,
                    title=doc.title,
                    document=doc.document,
                    score=doc.score,
                    semantic_rank=rank,
                    semantic_score=doc.score,
                )
            if (
                results[doc_id].semantic_rank < 0
                or results[doc_id].semantic_rank > rank
            ):
                results[doc_id].semantic_rank = rank

        rff_results: list[FormattedResults] = []
        for _, doc in results.items():
            bm25_rrf = rrf_score(doc.bm25_rank, k=k)
            sem_rrf = rrf_score(doc.semantic_rank, k=k)

            score = bm25_rrf + sem_rrf
            doc.rff_score = score
            rff_results.append(doc)
        rff_results = [x.model_dump() for x in rff_results]
        return sorted(rff_results, key=lambda x: x["rff_score"], reverse=True)[:limit]

    def _semantic_chunk_search(
        self, query: str, limit: int = DEFAULT_SEARCH_LIMIT
    ) -> List[FormattedResults]:
        return self.semantic_search.search_chunks(query, limit)


def rrf_score(rank, k=60) -> float:
    return 1 / (k + rank)


def semantic_chunk_search(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> dict:
    model = HybirdSearch(load_movies())
    return model._semantic_chunk_search(query, limit)


def normalise_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)

    if max_score == min_score:
        return [1.0] * len(scores)

    normalised_scores = []
    for s in scores:
        normalised_scores.append((s - min_score) / (max_score - min_score))
    return normalised_scores


def normalise_search_results(results: list[FormattedResults]) -> List[FormattedResults]:
    scores = []
    for res in results:
        scores.append(res.score)

    norm_score = normalise_scores(scores)
    for i, res in enumerate(results):
        res.norm_score = norm_score[i]

    return results


def hybrid_score(bm25_score: float, semantic_score: float, alpha: float = 0.5):
    return alpha * bm25_score + (1 - alpha) * semantic_score


def weighted_search(
    query: str, alpha: float = 0.5, limit: int = 5
) -> List[FormattedResults]:
    model = HybirdSearch(load_movies())
    return model.weighted_search(query, alpha, limit)


def rrf_search_command(
    query: str,
    k: int = 60,
    limit: int = 5,
    enhance: Optional[str] = None,
    rerank_method: Optional[str] = None,
) -> dict:
    documents = load_movies()
    model = HybirdSearch(documents)

    original_query = query
    enhanced_query = None
    if enhance:
        enhanced_query = enhance_query(query, method=enhance)
        query = enhanced_query

    search_limit = limit * SEARCH_MULTIPLIER if rerank_method else limit
    results = model.rrf_search(query, k, search_limit)

    if rerank_method:
        results = rerank_command(
            query, results, method=rerank_method, limit=search_limit
        )

    return {
        "original_query": original_query,
        "enhanced_query": enhanced_query,
        "enhance_method": enhance,
        "query": query,
        "k": k,
        "rerank_method": rerank_method,
        "results": results[:limit],
    }
