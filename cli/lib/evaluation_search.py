import json
from .hybird_search import HybirdSearch
from .search_utils import load_movies
from .llm_model import LLMModel


def precision_at_k(
    retrieved_docs: list[str], relevant_docs: set[str], k: int = 5
) -> float:
    top_k = retrieved_docs[:k]
    relevant_count = 0
    for doc in top_k:
        if doc in relevant_docs:
            relevant_count += 1
    return relevant_count / k


def recall_at_k(
    retrieved_docs: list[str], relevant_docs: set[str], k: int = 5
) -> float:
    top_k = retrieved_docs[:k]
    relevant_count = 0
    for doc in top_k:
        if doc in relevant_docs:
            relevant_count += 1
    return relevant_count / len(relevant_docs)


def f1_score(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)


def evaluation_search(eval_set: dict, limit: int = 5) -> list[dict]:
    results = []

    for test in eval_set["test_cases"]:

        query = test["query"]
        if query not in [
            "children's animated bear adventure",
            "friendship transformation magic with bears",
        ]:
            continue
        relevant_docs = test["relevant_docs"]
        hybrid_model = HybirdSearch(load_movies())
        rrf_scores = hybrid_model.rrf_search(query=query, k=60, limit=limit)[:limit]
        retrieved_docs = []
        for result in rrf_scores:
            title = result.get("title", "")
            if title:
                retrieved_docs.append(title)

        precision = precision_at_k(retrieved_docs, relevant_docs, limit)
        recall = recall_at_k(retrieved_docs, relevant_docs, limit)
        f1 = f1_score(precision, recall)

        results.append(
            {
                "query": query,
                "precision": precision,
                "retrieved": retrieved_docs,
                "relevant": relevant_docs,
                "recall": recall,
                "f1-score": f1,
            }
        )

    return results


def llm_evaluate(query: str, results: list[dict]) -> list[int]:
    MODEL = LLMModel(complex=True)
    prompt = f"""Rate how relevant each result is to this query on a 0-3 scale:

Query: "{query}"

Results:
{chr(10).join(results)}

Scale:
- 3: Highly relevant
- 2: Relevant
- 1: Marginally relevant
- 0: Not relevant

Do NOT give any numbers out than 0, 1, 2, or 3.

Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

[2, 0, 3, 2, 1]

Be sure not to return more scores than documents provided."""
    response = MODEL.generate_content(prompt)
    res = json.loads(response)
    return res
