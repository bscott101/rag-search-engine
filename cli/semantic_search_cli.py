import argparse

from lib.semantic_search import (
    verify_model,
    embed_text,
    verify_embeddings,
    embed_query_text,
    search_command,
    chunk_text,
)
from lib.search_utils import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify the model is able to be loaded")
    emb_text = subparsers.add_parser("embed_text", help="Embeds a text string")
    emb_text.add_argument("text", type=str, help="string you want to embed")

    verify_embs = subparsers.add_parser(
        "verify_embeddings", help="Verufy that the model can load and run embeddings"
    )

    embed_query = subparsers.add_parser("embedquery", help="embed query text")
    embed_query.add_argument("query", type=str, help="query that you wish to embed")

    embed_search = subparsers.add_parser(
        "search", help="Query to embed and search on documents"
    )
    embed_search.add_argument(
        "query", type=str, help="Query term that you want to search on documents"
    )
    embed_search.add_argument(
        "--limit", type=int, nargs="?", default=5, help="Number of results to return"
    )

    chunk = subparsers.add_parser("chunk", help="Chunks text to the chunck size")
    chunk.add_argument("text", type=str, help="Input text to chunk")
    chunk.add_argument(
        "--chunk-size",
        type=int,
        nargs="?",
        default=200,
        help="Number of characters that you want to chunks to be",
    )
    chunk.add_argument("--overlap", type=int, nargs="?", default=0, help="overlap")

    semantic_chunk = subparsers.add_parser(
        "semantic_chunk", help="Chunk text for semantic chunks"
    )
    semantic_chunk.add_argument(
        "text", type=str, help="Text to chunk for semantic processing"
    )
    semantic_chunk.add_argument(
        "--max-chunk-size",
        type=int,
        nargs="?",
        default=4,
        help="Max number of sentences for each chunk",
    )
    semantic_chunk.add_argument(
        "--overlap", type=int, nargs="?", default=0, help="overlap between sentences"
    )

    args = parser.parse_args()
    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embed_query":
            embed_query_text(args.query)
        case "search":
            results = search_command(args.query, args.limit)
            for index, result in enumerate(results):
                print(f"{index + 1}. {result['title']} (score: {result['score']})")
                print(f"\t{result['description']}")
        case "chunk":
            chunk_text(args.text, args.chunk_size, args.overlap, args.command)
        case "semantic_chunk":
            chunk_text(args.text, args.max_chunk_size, args.overlap, args.command)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
