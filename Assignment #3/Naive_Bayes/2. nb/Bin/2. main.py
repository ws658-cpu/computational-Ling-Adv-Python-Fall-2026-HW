import argparse

from nb.nb import NaiveBayes
from utils.load_data import build_dataframe


def main():
    parser = argparse.ArgumentParser(
        description="Predict speech authors using Naive Bayes."
    )
    parser.add_argument(
        "-f",
        "--indir",
        required=True,
        help="Folder containing kennedy, johnson, and unlabeled.",
    )
    args = parser.parse_args()

    training_df, test_df = build_dataframe(args.indir)

    model = NaiveBayes(alpha=0.1)
    model.train(training_df)
    predictions = model.test(test_df)

    print("Class labels: 0 = Kennedy, 1 = Johnson")
    print("Priors:", model.priors)
    print("Predictions:", predictions)


if __name__ == "__main__":
    main()
