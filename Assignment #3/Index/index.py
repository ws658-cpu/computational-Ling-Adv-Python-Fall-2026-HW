import argparse

parser = argparse.ArgumentParser(description="Build an inverted index.")

parser.add_argument(
    "-f",
    required=True,
    help="Path to the XML collection",
)

parser.add_argument(
    "-t",
    required=True,
    choices=["whitespace", "offtheshelf", "ngram"],
    help="Tokenizer to use",
)

parser.add_argument(
    "-s",
    required=True,
    choices=["yes", "no"],
    help="Whether to apply stemming",
)

args = parser.parse_args()

print("File:", args.f)
print("Tokenizer:", args.t)
print("Stemming:", args.s)
