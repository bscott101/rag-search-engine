from .hybird_search import HybirdSearch
from .llm_model import llm_inference
from .search_utils import DEFAULT_SEARCH_LIMIT, SEARCH_MULTIPLIER, load_movies


def generate_answer(query: str, results: list[dict], limit: int) -> str:
    docs = []
    for doc in results[:limit]:
        docs.append(f"{doc["title"]}: {doc["document"]}")

    prompt = f"""Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.

Query: {query}

Documents:
{docs}

Provide a comprehensive answer that addresses the query:"""
    res = llm_inference(prompt, max_new_tokens=2_000)

    return res


def generate_summary(query: str, results: list[dict], limit: int):
    docs = []
    for doc in results[:limit]:
        docs.append(f"{doc["title"]}: {doc["document"]}")

    prompt = f"""
Provide information useful to this query by synthesizing information from multiple search results in detail.
The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.
This should be tailored to Hoopla users. Hoopla is a movie streaming service.
Query: {query}
Search Results:
{docs}
Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:
"""
    res = llm_inference(prompt, max_new_tokens=2_000)

    return res


def rag(query: str, limit: int = DEFAULT_SEARCH_LIMIT):
    hybird_search = HybirdSearch(load_movies())
    search_results = hybird_search.rrf_search(
        query, k=60, limit=limit * SEARCH_MULTIPLIER
    )
    del hybird_search
    for doc in search_results:
        print(f'title: {doc["title"]}  rrf_score: {doc["rff_score"]}')

    response = generate_answer(query, search_results, limit)

    return {
        "query": query,
        "search_results": search_results[:limit],
        "response": response,
    }


def summarize(query: str, limit: int) -> dict:
    hybird_search = HybirdSearch(load_movies())
    search_results = hybird_search.rrf_search(
        query, k=60, limit=limit * SEARCH_MULTIPLIER
    )

    response = generate_summary(query, search_results, limit)

    return {
        "query": query,
        "search_results": search_results[:limit],
        "response": response,
    }


def generate_citation(query: str, results: list[dict], limit: int):
    context = ""
    for i, result in enumerate(results[:limit], start=1):
        context += f"[{i}]: {result['title']}; {result['document']}\n\n"

    prompt = f"""Answer the question or provide information based on the provided documents.

If not enough information is available to give a good answer, say so but give as good of an answer as you can while citing the sources you have.

Query: {query}

Documents:
{context}

Instructions:
- Provide a comprehensive answer that addresses the query
- Cite sources using [1], [2], etc. format when referencing information
- If sources disagree, mention the different viewpoints
- If the answer isn't in the documents, say "I don't have enough information"
- Be direct and informative

Answer:"""
    res = llm_inference(prompt, max_new_tokens=2_000)

    return res


def citations(query: str, limit: int = 5) -> dict:
    hybird_search = HybirdSearch(load_movies())
    search_result = hybird_search.rrf_search(
        query, k=60, limit=limit * SEARCH_MULTIPLIER
    )

    response = generate_citation(query, search_result, limit)

    return {
        "query": query,
        "search_results": search_result[:limit],
        "response": response,
    }


def generate_question(query: str, results: list[dict], limit: int) -> str:
    context = ""
    for i, result in enumerate(results[:limit], start=1):
        context += f"[{i}]: {result['title']}; {result['document'][:1_000]}\n\n"

    prompt = f"""Answer the user's question based on the provided movies.

Question: {query}

Documents:
{context}

Instructions:
- Answer questions directly and concisely
- Be casual and conversational
- Don't be cringe or hype-y
- Talk like a normal person would in a chat conversation

Answer:"""

    res = llm_inference(prompt, max_new_tokens=2_000)

    return res


def question(query: str, limit: int):
    hybird_search = HybirdSearch(load_movies())
    search_result = hybird_search.rrf_search(
        query, k=60, limit=limit * SEARCH_MULTIPLIER
    )

    response = generate_question(query, search_result, limit)

    return {
        "query": query,
        "search_results": search_result[:limit],
        "response": response,
    }
