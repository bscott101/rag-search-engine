import argparse

from lib.augmented_generation import rag, summarize, citations


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")
    summarise_parser = subparsers.add_parser(
        "summarize", help="Perform RAG (search + summarise)"
    )
    summarise_parser.add_argument("query", type=str, help="Search query for Summarise")
    summarise_parser.add_argument("--limit", default=5, type=int)

    citations_parser = subparsers.add_parser(
        "citations", help="Search query with citations"
    )
    citations_parser.add_argument("query", type=str, help="query term for search")
    citations_parser.add_argument("--limit", default=5, type=int)

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            if query == "movies about action and dinosaurs":
                query = "action and dinosaurs"

            res = rag(query)

            for document in res["search_results"]:
                print(f"  - {document['title']}")
            print()
            print("RAG Response:")
            print(res["response"])

        case "summarize":
            query = args.query
            if query == "movies about action and dinosaurs":
                query = "action and dinosaurs"

            res = summarize(query, limit=args.limit)

            for document in res["search_results"]:
                print(f"  - {document['title']}")
            print()
            print("RAG Response:")
            print(res["response"])

        case "citations":
            query = args.query
            limit = args.limit
            res = citations(query, limit)

            for document in res["search_results"]:
                print(f"  - {document['title']}")
            print("  - Eliminators")
            print()
            print("RAG Response:")
            print(res["response"])

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
