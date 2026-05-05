from .llm_model import llm_inference
from typing import Optional


def spell_correct(query: str) -> str:
    prompt = f"""Fix any spelling errors in this movie search query.

Only correct obvious typos. Don't change correctly spelled words.

Query: "{query}"

If no errors, return the original query.
Corrected:"""

    return llm_inference(prompt)

def rewrite_query(query: str) -> str:
    prompt = f"""Rewrite this movie search query to be more specific and searchable.

Original: "{query}"

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep it concise (under 10 words)
- It should be a google style search query that's very specific
- Don't use boolean logic

Examples:

- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

Rewritten query:"""
    # response = client.models.generate_content(model=model, contents=prompt)
    return llm_inference(prompt)


def expand_query(query: str) -> str:
    prompt = f"""Expand this movie search query with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
This will be appended to the original query.

Examples:

- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
- "action movie with bear" -> "action thriller bear chase fight adventure"
- "comedy with bear" -> "comedy funny bear humor lighthearted"

Query: "{query}"
"""
    return llm_inference(prompt)


def enhance_query(query: str, method: Optional[str] = None) -> str:
    match method:
        case "spell":
            return spell_correct(query)
        case "rewrite":
            return rewrite_query(query)
        case "expand":
            return expand_query(query)
        case _:
            return query


def rerank_query(query: str, doc: dict) -> str:
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

    return llm_inference(prompt)


def rerank_method(query: str, results: list[dict]) -> list[dict]:
    for doc in results:
        score = rerank_query(query, doc)
        doc["rerank_score"] = float(score)

    return sorted(results, key=lambda x: x["rerank_score"], reverse=True)
