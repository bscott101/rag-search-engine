import argparse

from lib.keyword_search import build_command, idx_command, search_command, tf_command


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

    subparsers.add_parser("build", help="Create Index of Movies")

    args = parser.parse_args()
    match args.command:
        case "search":
            # print the search query here
            print(f"Searching for: {args.query}")
            movies = search_command(args.query)
            for index, item in enumerate(movies[:5]):
                print(f"{index + 1}. {item.title}")
        case "tf":
            count = tf_command(args.doc_id, args.query)
            print(count)

        case "idf":
            idf = idx_command(args.query)
            print(f"Inverse document frequency of '{args.query}': {idf:.2f}")

        case "build":
            print("Building index")
            build_command()
            print("Sucessfully built and saved")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
