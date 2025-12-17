import argparse


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    args = parser.parse_args()

    match args.command:
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
