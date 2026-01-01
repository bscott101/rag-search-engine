import argparse

from lib.semantic_search import (
    verify_model,
    embed_text,
    verify_embeddings,
    embed_query_text,
    search_command,
    chunk_text,
    embed_chunks,
    semantic_chunk_search,
)
from lib.search_utils import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_SEMANTIC_LIMIT,
)


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
    subparsers.add_parser("embed_chunks", help="Create semantic embedding chunks")

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
        case "embed_chunks":
            embeddings = embed_chunks()
            print(f"Generated {len(embeddings)} chunked embeddings")
        case "search_chunked":
            query_result = semantic_chunk_search(args.query, args.limit)
            result = query_result["result"]
            for i, res in enumerate(result):
                print(f"\n{i + 1}. {res["title"]} (score: {res['score']:.4f})")
                print(f"   {res["document"]}...")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
