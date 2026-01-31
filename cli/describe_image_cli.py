import argparse
from pathlib import Path

from lib.llm_model import LLMModel


def main():
    parser = argparse.ArgumentParser(description="Describe Image CLI")
    parser.add_argument("--image", type=Path, help="path to file")
    parser.add_argument("--query", type=str, help="query to rewrite based on the image")

    args = parser.parse_args()
    response = describe_image_command(str(args.image), args.query)

    print(f"Rewritten query: {response}")
    print("Total tokens:")



def describe_image_command(image_path: str, query: str):
    MODEL = LLMModel()
    sys_prompt = """Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
- Synthesize visual and textual information
- Focus on movie-specific details (actors, scenes, style, etc.)
- Return only the rewritten query, without any additional commentary"""

    return MODEL.generate_content(
        prompt=query, input_image=image_path, system_prompt=sys_prompt
    )

if __name__ == "__main__":
    main()