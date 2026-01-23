import json
import argparse
from lib.search_utils import EVAL_PATH, SCORE_PRECISION
from lib.evaluation_search import evaluation_search


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit
    with open(EVAL_PATH, "r") as file:
        eval_data_set = json.load(file)
    results = evaluation_search(eval_data_set, limit, args.top_k)

    print(f"k={limit}\n")
    for test_case in results:
        print(f"- Query: {test_case["query"]}")
        print(f"  - Precision@{limit}: {test_case["precision"]:.4f}")
        print(f"  - Recall@{limit}: {test_case["recall"]:.4f}")
        print(f"  - Retrieved: {test_case["retrieved"]}")
        print(f"  - Relevant: {test_case["relevant"]}")


if __name__ == "__main__":
    main()
