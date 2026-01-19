import json
from .llm_model import LLMModel


def individual_query(query: str, doc: dict, MODEL: LLMModel) -> str:
    prompt = f"""Rate how well this movie matches the search query.

Query: "{query}"
Movie: {doc.get("title", "")} - {doc.get("document", "")}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Give me ONLY the number in your response, no other text or explanation.

Score:"""

    return MODEL.generate_content(prompt)


def rerank_individual(query: str, results: list[dict], MODEL: LLMModel) -> list[dict]:
    for doc in results:
        score = individual_query(query, doc, MODEL)
        doc["rerank_score"] = float(score)

    return sorted(results, key=lambda x: x["rerank_score"], reverse=True)


def rerank_batch(query: str, documents: list[dict], MODEL: LLMModel) -> list[dict]:
    doc_list_str = []

    for doc in documents:
        print(f'id: {id} title: {doc["title"]}')
        doc_list_str.append(
            f"ID: {doc["doc_id"]} Title: {doc['title']} Description: {doc['document'][:400]}"
        )

    prompt = f"""Rank these movies by relevance to the search query.

Query: "{query}"

Movies:
{doc_list_str}

Return ONLY the IDs in order of relevance (best match first). Return a valid JSON list, nothing else. For example:

[1, 3, 4, 2]
"""
    # surely having an AI rank other AI items is not a good idea
    response = MODEL.generate_content(
        prompt,
        system_prompt="You are a movie critic",
    )
    res = json.loads(response)

    if len(res) > len(documents):
        res = res[: len(documents)]

    if len(res) < len(documents):
        diff = len(documents) - len(res)
        res.extend([9999 for _ in range(0, diff)])

    for id, score in enumerate(res):
        documents[id]["rerank_batch"] = score

    return sorted(documents, key=lambda x: x["rerank_batch"])


def rerank_command(
    query: str, documents: list[dict], method: str = "batch", limit: int = 5
) -> list[dict]:
    MODEL = LLMModel(complex=True)

    if method == "individual":
        return rerank_individual(query, documents, limit, MODEL)

    if method == "batch":
        return rerank_batch(query, documents, limit, MODEL)

    return documents[:limit]
