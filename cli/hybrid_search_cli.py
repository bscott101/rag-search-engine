import argparse

from lib.hybird_search import (
    semantic_chunk_search,
    normalize_scores,
    weighted_search,
    rrf_search_command,
)
from lib.search_utils import DEFAULT_SEMANTIC_LIMIT


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

    args = parser.parse_args()
    match args.command:
        case "search_chunked":
            query_result = semantic_chunk_search(args.query, args.limit)
            result = query_result["result"]
            for i, res in enumerate(result):
                print(f"\n{i + 1}. {res["title"]} (score: {res['score']:.4f})")
                print(f"   {res["document"]}...")

        case "normalize":
            scores = normalize_scores(args.scores)
            for score in scores:
                print(f"* {score:.4f}")

        case "weighted-search":
            result = weighted_search(args.query, args.alpha, args.limit)
            for index, res in enumerate(result):
                print(f"{index + 1}. {res['title']}")
                print(f"    Hybrid Score: {res['hybrid']}")
                print(f"    BM25: {res['bm25']}, Semantic: {res['semantic']}")
                print(f"    {res['document']}...")

        case "rrf-search":
            result = rrf_search_command(args.query, args.k, args.limit, args.enhance)

            if result["enhanced_query"]:
                print(
                    f"Enhanced query ({result['enhance_method']}): '{result['original_query']}' -> '{result['enhanced_query']}'\n"
                )

            for index, res in enumerate(result["results"], start=1):
                print(f"{index}. {res['title']}")
                print(f"    RRF Score: {res['rrf_score']}")
                print(
                    f"    BM25 Rank: {res['bm25_rank']}, Semantic Rank: {res['semantic_rank']}"
                )
                print(f"    {res['document']}...")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
