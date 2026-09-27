import argparse
from collections import Counter
from matplotlib import pyplot as plt


def get_ranks_and_frequencies(infile):
    """Produces a list of rank, frequency pairs for each word in a text file
    :param infile: a text file
    :return: a list containing rank, frequency pairs for each word
    """
    with open(infile,encoding="utf-8") as f:
        contents = f.read()
    c = Counter(contents.split())

    frequencies = sorted(c.values(), reverse=True)
    ranks_and_frequencies = list(enumerate(frequencies, start=1))
    
    return ranks_and_frequencies


def plot(infile):
    """
    Plots rank and frequency pairs to demonstrate Zipf's Law
    :param infile: a text file
    :return: None, produces a matplotlib plot
    """
    ranks_and_frequencies = get_ranks_and_frequencies(infile)

    ranks = [rank for rank, frequency in ranks_and_frequencies]
    frequencies = [frequency for rank, frequency in ranks_and_frequencies]

    plt.plot(ranks, frequencies)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Rank")
    plt.ylabel("Frequency")
    plt.title("Whitney Syriaque")
    plt.savefig("graph.png")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Constructs a curve '
                                                 'demonstrating Zipf\'s Law '
                                                 'by plotting a rank, '
                                                 'frequency plot.')
    parser.add_argument('--path', type=str, required=True, help='Path to file')
    args = parser.parse_args()
    plot(args.path)
