from .hybird_search import rrf_search_command


def evaluation_search(eval_set: dict, limit: int = 60, k: int = 60) -> list[dict]:
    results = []

    for test in eval_set["test_cases"]:

        query = test["query"]
        if query not in ["car racing", "cute british bear marmalade"]:
            continue
        relevant_docs = test["relevant_docs"]
        rrf_scores = rrf_search_command(query=query, limit=limit)
        retrieved = [x["title"] for x in rrf_scores["results"]]
        hits = [x for x in retrieved if x in relevant_docs]
        precision = len(hits) / len(retrieved)
        recall = len(hits) / len(relevant_docs)

        results.append(
            {
                "query": query,
                "precision": precision,
                "retrieved": retrieved,
                "relevant": relevant_docs,
                "recall": recall,
            }
        )

    return results
