import argparse

from lib import keyword_search


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    tf_parser = subparsers.add_parser(
        "tf", help="Search the count of words for a movie"
    )
    tf_parser.add_argument(
        "doc_id", type=int, help="Document Id for word count frequency"
    )
    tf_parser.add_argument("query", type=str, help="Word Frequency")

    idf_parser = subparsers.add_parser("idf", help="Inverse Document Frequency")
    idf_parser.add_argument("query", type=str, help="Token search")

    tf_idf = subparsers.add_parser(name="tfidf", help="Get TF IDF score")
    tf_idf.add_argument("doc_id", type=int, help="document id for scoring")
    tf_idf.add_argument("query", type=str, help="word for scoring")

    subparsers.add_parser("build", help="Create Index of Movies")

    args = parser.parse_args()
    match args.command:
        case "search":
            # print the search query here
            print(f"Searching for: {args.query}")
            movies = keyword_search.search_command(args.query)
            for index, item in enumerate(movies[:5]):
                print(f"{index + 1}. {item.title}")
        case "tf":
            count = keyword_search.tf_command(args.doc_id, args.query)
            print(count)

        case "idf":
            idf = keyword_search.idx_command(args.query)
            print(f"Inverse document frequency of '{args.query}': {idf:.2f}")

        case "tfidf":
            tf_idf = keyword_search.tfidf_command(args.doc_id, args.query)
            print(
                f"TF-IDF score of '{args.query}' in document '{args.doc_id}': {tf_idf:.2f}"
            )

        case "build":
            print("Building index")
            keyword_search.build_command()
            print("Sucessfully built and saved")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
