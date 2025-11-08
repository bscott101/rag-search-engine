import argparse
from lib.keyword_search import search_command, build_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Create Index of Movies")

    args = parser.parse_args()
    match args.command:
        case "search":
            # print the search query here
            print(f"Searching for: {args.query}")

            movies = search_command(args.query)
            for index, item in enumerate(movies[:5]):
                print(f"{index + 1}. {item.title}")
        case "build":
            print("Building index")
            build_command()
            print("Sucessfully built and saved")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
