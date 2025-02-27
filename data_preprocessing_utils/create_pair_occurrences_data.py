"""Create occurrence statistics for a given concept pair"""
import argparse
import json
import os
import pickle
import sys

from tqdm import tqdm

from utils import (
    PAIR_OCCURENCES,
    ADJECTIVE_OCCURRENCES,
    NOUN_OCCURRENCES,
    NOUNS,
    ADJECTIVES,
    contains_adjective_noun_pair,
    OCCURRENCE_DATA,
    DATA_COCO_SPLIT,
    POS_TAGGED_CAPTIONS_FILENAME,
    VERBS,
    contains_verb_noun_pair,
    VERB_OCCURRENCES,
)


def count_adjective_noun_pairs(nouns_file, adjectives_file, preprocessed_data_folder):
    with open(nouns_file, "r") as json_file:
        nouns = json.load(json_file)

    with open(adjectives_file, "r") as json_file:
        adjectives = json.load(json_file)

    with open(
        os.path.join(preprocessed_data_folder, POS_TAGGED_CAPTIONS_FILENAME), "rb"
    ) as pickle_file:
        captions = pickle.load(pickle_file)

    first_noun = nouns[0]
    first_adjective = adjectives[0]

    print("Looking for pairs: {} - {}".format(adjectives, nouns))

    data = {}
    data[NOUNS] = nouns
    data[ADJECTIVES] = adjectives

    occurrence_data = {}

    for coco_id, tagged_caption in tqdm(captions.items()):
        occurrence_data[coco_id] = {}
        occurrence_data[coco_id][PAIR_OCCURENCES] = 0
        occurrence_data[coco_id][ADJECTIVE_OCCURRENCES] = 0
        occurrence_data[coco_id][NOUN_OCCURRENCES] = 0
        occurrence_data[coco_id][DATA_COCO_SPLIT] = tagged_caption[DATA_COCO_SPLIT]

        for caption in tagged_caption["pos_tagged_captions"]:
            noun_is_present, adjective_is_present, combination_is_present = contains_adjective_noun_pair(
                caption, nouns, adjectives
            )
            if combination_is_present:
                print(" ".join([token.text for token in caption.tokens]))
                occurrence_data[coco_id][PAIR_OCCURENCES] += 1
            if adjective_is_present:
                occurrence_data[coco_id][ADJECTIVE_OCCURRENCES] += 1
            if noun_is_present:
                occurrence_data[coco_id][NOUN_OCCURRENCES] += 1

    data[OCCURRENCE_DATA] = occurrence_data

    data_path = "{}_{}.json".format(first_adjective, first_noun)
    print("\nSaving results to {}".format(data_path))
    with open(data_path, "w") as json_file:
        json.dump(data, json_file)

    for n in range(1, 6):
        noun_occurences = len(
            [d for d in occurrence_data.values() if d[NOUN_OCCURRENCES] >= n]
        )
        adjective_occurences = len(
            [d for d in occurrence_data.values() if d[ADJECTIVE_OCCURRENCES] >= n]
        )
        pair_occurences = len(
            [d for d in occurrence_data.values() if d[PAIR_OCCURENCES] >= n]
        )

        print(
            "\nFound {}\timages where the noun occurs at least {} time(s).".format(
                noun_occurences, n
            )
        )
        print(
            "Found {}\timages where the adjective occurs at least {} time(s).".format(
                adjective_occurences, n
            )
        )
        print(
            "Found {}\timages where the pair occurs at least {} time(s).".format(
                pair_occurences, n
            )
        )


def count_verb_noun_pairs(nouns_file, verbs_file, preprocessed_data_folder):
    with open(nouns_file, "r") as json_file:
        nouns = json.load(json_file)

    with open(verbs_file, "r") as json_file:
        verbs = json.load(json_file)

    with open(
        os.path.join(preprocessed_data_folder, POS_TAGGED_CAPTIONS_FILENAME), "rb"
    ) as pickle_file:
        captions = pickle.load(pickle_file)

    first_noun = nouns[0]
    first_verb = verbs[0]

    print("Looking for pairs: {} - {}".format(verbs, nouns))

    data = {}
    data[NOUNS] = nouns
    data[VERBS] = verbs

    occurrence_data = {}

    for coco_id, tagged_caption in tqdm(captions.items()):
        occurrence_data[coco_id] = {}
        occurrence_data[coco_id][PAIR_OCCURENCES] = 0
        occurrence_data[coco_id][VERB_OCCURRENCES] = 0
        occurrence_data[coco_id][NOUN_OCCURRENCES] = 0
        occurrence_data[coco_id][DATA_COCO_SPLIT] = tagged_caption[DATA_COCO_SPLIT]

        for caption in tagged_caption["pos_tagged_captions"]:
            noun_is_present, verb_is_present, combination_is_present = contains_verb_noun_pair(
                caption, nouns, verbs
            )
            if combination_is_present:
                print(" ".join([token.text for token in caption.tokens]))
                occurrence_data[coco_id][PAIR_OCCURENCES] += 1
            if verb_is_present:
                occurrence_data[coco_id][VERB_OCCURRENCES] += 1
            if noun_is_present:
                occurrence_data[coco_id][NOUN_OCCURRENCES] += 1

    data[OCCURRENCE_DATA] = occurrence_data

    data_path = "{}_{}.json".format(first_verb, first_noun)
    print("\nSaving results to {}".format(data_path))
    with open(data_path, "w") as json_file:
        json.dump(data, json_file)

    for n in range(1, 6):
        noun_occurences = len(
            [d for d in occurrence_data.values() if d[NOUN_OCCURRENCES] >= n]
        )
        verb_occurences = len(
            [d for d in occurrence_data.values() if d[VERB_OCCURRENCES] >= n]
        )
        pair_occurences = len(
            [d for d in occurrence_data.values() if d[PAIR_OCCURENCES] >= n]
        )

        print(
            "\nFound {}\timages where the noun occurs at least {} time(s).".format(
                noun_occurences, n
            )
        )
        print(
            "Found {}\timages where the verb occurs at least {} time(s).".format(
                verb_occurences, n
            )
        )
        print(
            "Found {}\timages where the pair occurs at least {} time(s).".format(
                pair_occurences, n
            )
        )


def check_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nouns",
        help="Path to file containing JSON-serialized list of nouns (target noun + synonyms).",
        required=True,
    )
    parser.add_argument(
        "--adjectives",
        help="Path to file containing JSON-serialized list of adjectives (target adjective + synonyms)",
    )
    parser.add_argument(
        "--verbs",
        help="Path to file containing JSON-serialized list of verbs (target verb + synonyms)",
    )
    parser.add_argument(
        "--preprocessed-data-folder",
        help="Folder where the preprocessed data is located",
        default="../datasets/coco2014_preprocessed/",
    )

    parsed_args = parser.parse_args(args)
    print(parsed_args)
    return parsed_args


if __name__ == "__main__":
    parsed_args = check_args(sys.argv[1:])
    if parsed_args.adjectives and parsed_args.verbs:
        raise ValueError("Specify either verbs or adjectives file, not both!")
    if not parsed_args.adjectives and not parsed_args.verbs:
        raise ValueError("Specify either verbs or adjectives file!")

    if parsed_args.adjectives:
        count_adjective_noun_pairs(
            parsed_args.nouns,
            parsed_args.adjectives,
            parsed_args.preprocessed_data_folder,
        )
    elif parsed_args.verbs:
        count_verb_noun_pairs(
            parsed_args.nouns, parsed_args.verbs, parsed_args.preprocessed_data_folder
        )
