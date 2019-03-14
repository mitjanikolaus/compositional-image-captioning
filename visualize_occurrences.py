import argparse
import json
import sys
import numpy as np
import matplotlib.pyplot as plt

from utils import (
    OCCURRENCE_DATA,
    PAIR_OCCURENCES,
    NOUN_OCCURRENCES,
    ADJECTIVE_OCCURRENCES,
)


def visualize_occurrences(occurrences_data_file):
    with open(occurrences_data_file, "r") as json_file:
        occurrences_data = json.load(json_file)

    pair_matches = np.zeros(5)
    noun_matches = np.zeros(5)
    adjective_matches = np.zeros(5)
    for n in range(len(pair_matches)):
        noun_matches[n] = len(
            [
                key
                for key, value in occurrences_data[OCCURRENCE_DATA].items()
                if value[NOUN_OCCURRENCES] > n
            ]
        )
        adjective_matches[n] = len(
            [
                key
                for key, value in occurrences_data[OCCURRENCE_DATA].items()
                if value[ADJECTIVE_OCCURRENCES] > n
            ]
        )
        pair_matches[n] = len(
            [
                key
                for key, value in occurrences_data[OCCURRENCE_DATA].items()
                if value[PAIR_OCCURENCES] > n
            ]
        )

    print("Noun matches:")
    for n in range(len(pair_matches)):
        print(str(noun_matches[n]) + " | ", end="")
    print("\nAdjective matches:")
    for n in range(len(pair_matches)):
        print(str(adjective_matches[n]) + " | ", end="")
    print("\nPair matches:")
    for n in range(len(pair_matches)):
        print(str(pair_matches[n]) + " | ", end="")
    print("\n")

    patches, texts, _ = plt.pie(pair_matches, autopct="%1.2f")
    labels = ["N=1", "N=2", "N=3", "N=4", "N=5"]
    plt.legend(patches, labels, loc="best")
    plt.axis("equal")
    plt.tight_layout()
    plt.show()


def check_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--occurrences-data",
        help="File containing occurrences statistics about adjective noun pairs",
        default="data/brown_dog.json",
    )

    parsed_args = parser.parse_args(args)
    print(parsed_args)
    return parsed_args


if __name__ == "__main__":
    parsed_args = check_args(sys.argv[1:])
    visualize_occurrences(occurrences_data_file=parsed_args.occurrences_data)
