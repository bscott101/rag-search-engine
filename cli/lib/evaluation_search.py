from .hybird_search import HybirdSearch
from .search_utils import load_movies


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


def evaluation_search(eval_set: dict, limit: int = 60, k: int = 60) -> list[dict]:
    results = []

    for test in eval_set["test_cases"]:

        query = test["query"]
        if query not in [
            "cute british bear marmalade",
            "children's animated bear adventure",
            "friendship transformation magic with bears",
        ]:
            continue
        relevant_docs = test["relevant_docs"]
        hybrid_model = HybirdSearch(load_movies())
        rrf_scores = hybrid_model.rrf_search(query=query, k=k, limit=limit)
        retrieved_docs = []
        for result in rrf_scores:
            title = result.get("title", "")
            if title:
                retrieved_docs.append(title)

        precision = precision_at_k(retrieved_docs, relevant_docs, limit)
        recall = recall_at_k(retrieved_docs, relevant_docs, limit)

        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        f1_score = 2 * (precision * recall) / (precision + recall)

        results.append(
            {
                "query": query,
                "precision": precision,
                "retrieved": retrieved_docs,
                "relevant": relevant_docs,
                "recall": recall,
                "f1-score": f1_score,
            }
        )

    return results
