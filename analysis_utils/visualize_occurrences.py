"""Visualize occurrence statistics for a concept pair"""

import argparse
import json
import os
import sys
import numpy as np

from utils import (
    OCCURRENCE_DATA,
    PAIR_OCCURENCES,
    NOUN_OCCURRENCES,
    ADJECTIVE_OCCURRENCES,
    ADJECTIVES,
    VERBS,
    VERB_OCCURRENCES,
    DATA_COCO_SPLIT,
)


def visualize_occurrences(occurrences_data_files):
    for coco_split in ["train2014", "val2014"]:
        print("Split: {}".format(coco_split))
        for occurrences_data_file in occurrences_data_files:
            with open(occurrences_data_file, "r") as json_file:
                occurrences_data = json.load(json_file)

            pair_matches = np.zeros(5)
            noun_matches = np.zeros(5)
            adjective_matches = np.zeros(5)
            verb_matches = np.zeros(5)

            for n in range(len(pair_matches)):
                noun_matches[n] = len(
                    [
                        key
                        for key, value in occurrences_data[OCCURRENCE_DATA].items()
                        if value[NOUN_OCCURRENCES] == n + 1
                        and value[DATA_COCO_SPLIT] == coco_split
                    ]
                )
                if ADJECTIVES in occurrences_data:
                    adjective_matches[n] = len(
                        [
                            key
                            for key, value in occurrences_data[OCCURRENCE_DATA].items()
                            if value[ADJECTIVE_OCCURRENCES] == n + 1
                            and value[DATA_COCO_SPLIT] == coco_split
                        ]
                    )
                if VERBS in occurrences_data:
                    verb_matches[n] = len(
                        [
                            key
                            for key, value in occurrences_data[OCCURRENCE_DATA].items()
                            if value[VERB_OCCURRENCES] == n + 1
                            and value[DATA_COCO_SPLIT] == coco_split
                        ]
                    )
                pair_matches[n] = len(
                    [
                        key
                        for key, value in occurrences_data[OCCURRENCE_DATA].items()
                        if value[PAIR_OCCURENCES] == n + 1
                        and value[DATA_COCO_SPLIT] == coco_split
                    ]
                )

            noun_name = (
                os.path.basename(occurrences_data_file).split("_")[1].split(".")[0]
            )
            print("\n" + noun_name, end=" | ")
            for n in range(len(pair_matches)):
                print(noun_matches[n], end=" | ")
            print(np.sum(noun_matches), end="")

            if ADJECTIVES in occurrences_data:
                adjective_name = os.path.basename(occurrences_data_file).split("_")[0]
                print("\n" + adjective_name, end=" | ")
                for n in range(len(pair_matches)):
                    print(adjective_matches[n], end=" | ")
                print(np.sum(adjective_matches), end="")

            if VERBS in occurrences_data:
                verb_name = os.path.basename(occurrences_data_file).split("_")[0]
                print("\n" + verb_name, end=" | ")
                for n in range(len(pair_matches)):
                    print(verb_matches[n], end=" | ")
                print(np.sum(verb_matches), end="")

            name = os.path.basename(occurrences_data_file).split(".")[0]
            print("\n" + name, end=" | ")
            for n in range(len(pair_matches)):
                print(pair_matches[n], end=" | ")
            print(np.sum(pair_matches), end="")

        print("\n")


def check_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--occurrences-data",
        nargs="+",
        help="Files containing occurrences statistics about adjective-noun or verb-noun pairs",
        required=True,
    )

    parsed_args = parser.parse_args(args)
    print(parsed_args)
    return parsed_args


if __name__ == "__main__":
    parsed_args = check_args(sys.argv[1:])
    visualize_occurrences(occurrences_data_files=parsed_args.occurrences_data)
