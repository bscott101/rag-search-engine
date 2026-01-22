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
    results = [x.model_dump() for x in results]
    for doc in results:
        score = individual_query(query, doc, MODEL)
        doc["individual_score"] = float(score)

    return sorted(results, key=lambda x: x["individual_score"], reverse=True)


def rerank_batch(query: str, documents: list[dict], MODEL: LLMModel) -> list[dict]:
    doc_list_str = []
    doc_map = {}
    for doc in documents:
        doc_map[doc["doc_id"]] = doc
        doc_list_str.append(
            f"ID: {doc["doc_id"]} Title: {doc['title']} Description: {doc['document'][:200]}"
        )
    prompt = f"""Rank these movies by relevance to the search query.

Query: "{query}"

Movies:
{doc_list_str}

Return ONLY the IDs in order of relevance (best match first). Return a valid JSON list, nothing else. For example:

[1, 3, 4, 2]

Just return the Raw List and nothing else.
"""
    # surely having an AI rank other AI items is not a good idea
    response = MODEL.generate_content(
        prompt,
        system_prompt="You are a movie critic",
    )
    results = []
    res = json.loads(response)
    for rank, doc_id in enumerate(res, start=1):
        try:
            doc = doc_map[doc_id]
            doc["rerank_batch"] = rank
            results.append(doc)
        except Exception as e:
            print(f"Error: {e}")

    return sorted(results, key=lambda x: x["rerank_batch"])


def rerank_command(
    query: str, documents: list[dict], method: str = "batch", limit: int = 5
) -> list[dict]:
    MODEL = LLMModel(complex=True)

    if method == "individual":
        return rerank_individual(query, documents[:limit], MODEL)

    if method == "batch":
        return rerank_batch(query, documents[:limit], MODEL)

    return documents
