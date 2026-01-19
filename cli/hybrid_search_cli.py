import argparse

from lib.hybird_search import (
    rrf_search_command,
    semantic_chunk_search,
    weighted_search,
    normalise_scores,
)
from lib.search_utils import DEFAULT_SEMANTIC_LIMIT, DOCUMENT_PREVIEW_LENGTH
from lib.schemas import FormattedResults


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_chunked = subparsers.add_parser(
        "search_chunked", help="Enter a query to search semanticly"
    )
    search_chunked.add_argument(
        "query",
        type=str,
        help="The query that you want to perform a embedded search on",
    )
    search_chunked.add_argument(
        "--limit",
        type=int,
        nargs="?",
        default=DEFAULT_SEMANTIC_LIMIT,
        help="Amount of document hits to return",
    )

    normalize = subparsers.add_parser(
        "normalize", help="Enter a list of scores for normalization"
    )
    normalize.add_argument(
        "scores", nargs="+", type=float, help="creates a list of scores"
    )

    weighted_search_command = subparsers.add_parser(
        "weighted-search", help="Query that uses both bm25 and chunked semantic search"
    )
    weighted_search_command.add_argument("query", type=str, help="Query for search")
    weighted_search_command.add_argument(
        "--alpha",
        nargs="?",
        type=float,
        default=0.5,
        help="Configure value for weight between bm25 and semantic search",
    )
    weighted_search_command.add_argument(
        "--limit", type=int, nargs="?", default=5, help="Number of results to return"
    )

    rff_command = subparsers.add_parser("rrf-search", help="Recipical Rank fusion")
    rff_command.add_argument("query", type=str, help="Query for the search")
    rff_command.add_argument(
        "-k",
        type=int,
        default=60,
        nargs="?",
        help="K: parameter to control the weight to top-ranked results",
    )
    rff_command.add_argument(
        "--limit", type=int, nargs="?", help="Number of results to return", default=5
    )
    rff_command.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite", "expand"],
        help="Query enhancment method",
    )
    rff_command.add_argument(
        "--rerank-method",
        help="Method to rerank the returned query with LLMs",
        type=str,
        choices=["individual", "batch"],
    )

    args = parser.parse_args()
    match args.command:
        case "search_chunked":
            query_result = semantic_chunk_search(args.query, args.limit)
            result = query_result["result"]
            for i, res in enumerate(result):
                print(f"\n{i + 1}. {res["title"]} (score: {res['score']:.4f})")
                print(f"   {res["document"]}...")

        case "normalize":
            scores = normalise_scores(args.scores)
            for score in scores:
                print(f"* {score:.4f}")

        case "weighted-search":
            result = weighted_search(args.query, args.alpha, args.limit)
            for index, res in enumerate(result[: args.limit]):
                print(f"{index + 1}. {res.title}")
                print(f"    Hybrid Score: {res.hybird_score}")
                print(f"    BM25: {res.bm25_score}, Semantic: {res.semantic_score}")
                print(f"    {res.document[:DOCUMENT_PREVIEW_LENGTH]}...")

        case "rrf-search":
            limit = args.limit
            if args.rerank_method:
                limit = limit * 5
            result = rrf_search_command(
                args.query, k=args.k, limit=limit, enhance=args.enhance
            )

            if result["enhanced_query"]:
                print(
                    f"Enhanced query ({result['enhance_method']}): '{result['original_query']}' -> '{result['enhanced_query']}'\n"
                )
            match args.rerank_method:
                case "individual":
                    new_result = rerank_method(
                        result["original_query"], result["results"]
                    )
                    for index, res in enumerate(new_result, start=1):
                        print(f"{index}. {res['title']}")
                        print(f"    Rerank Score: {res['rerank_score']}/10")
                        print(f"    RRF Score: {res['rrf_score']}")
                        print(
                            f"    BM25 Rank: {res['bm25_rank']}, Semantic Rank: {res['semantic_rank']}"
                        )
                        print(f"    {res['document'][:DOCUMENT_PREVIEW_LENGTH]}...")
                case "batch":
                    new_result = rerank_method(
                        result["original_query"], result["results"]
                    )
                    new_result = rerank_batch_method(
                        result["original_query"], new_result
                    )

                    for index, res in enumerate(new_result[: args.limit], start=1):
                        print(f"{index}. {res['title']}")
                        print(f"    Rerank Score: {res['rerank_batch']}")
                        print(f"    RRF Score: {res['rrf_score']}")
                        print(
                            f"    BM25 Rank: {res['bm25_rank']}, Semantic Rank: {res['semantic_rank']}"
                        )
                        print(f"    {res['document'][:DOCUMENT_PREVIEW_LENGTH]}...")

                case _:
                    for index, res in enumerate(result["results"], start=1):
                        print(f"{index}. {res['title']}")
                        print(f"    RRF Score: {res['rrf_score']}")
                        print(
                            f"    BM25 Rank: {res['bm25_rank']}, Semantic Rank: {res['semantic_rank']}"
                        )
                        print(f"    {res['document'][:DOCUMENT_PREVIEW_LENGTH]}...")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
