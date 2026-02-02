import argparse
from lib.multimodel_search import Clip
from lib.search_utils import DOCUMENT_PREVIEW_LENGTH
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    image_embedding = subparsers.add_parser("verify_image_embedding", help="verify model generates embedding")
    image_embedding.add_argument("image", type=Path, help="input image path")

    image_search = subparsers.add_parser("image_search", help="Search documents tahs meet ")
    image_search.add_argument("image", type=Path, help="input image path")

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            model = Clip()
            emb = model.embed_image(args.image)

            print(f"Embedding shape: {emb.shape[0]} dimensions")

        case "image_search":
            model = Clip()
            result = model.search_with_image(args.image)
            
            for i, res in enumerate(result[:3], start=1):
                print(f"{i}. {res['title']} (similarity: {str(res['score'])[:5]})" )
                print(f"   {res['description'][:DOCUMENT_PREVIEW_LENGTH]}...")


if __name__ == "__main__":
    main()