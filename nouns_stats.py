from collections import Counter

import argparse
import json
import os
import pickle
import sys

from tqdm import tqdm

from utils import (
    WORD_MAP_FILENAME,
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
    get_adjectives_for_noun,
)


def noun_stats(nouns_file, preprocessed_data_folder):
    with open(nouns_file, "r") as json_file:
        nouns = json.load(json_file)

    word_map_path = os.path.join(preprocessed_data_folder, WORD_MAP_FILENAME)
    with open(word_map_path, "r") as json_file:
        word_map = json.load(json_file)

    with open(
        os.path.join(preprocessed_data_folder, POS_TAGGED_CAPTIONS_FILENAME), "rb"
    ) as pickle_file:
        captions = pickle.load(pickle_file)

    nouns = {noun for noun in nouns if noun in word_map}

    print("Noun stats for: {}".format(nouns))

    adjective_frequencies = Counter()

    for coco_id, tagged_caption in tqdm(captions.items()):
        for caption in tagged_caption["pos_tagged_captions"]:
            adjectives = get_adjectives_for_noun(caption, nouns)
            if len(adjectives) == 0:
                adjective_frequencies["No adjective"] += 1
            adjective_frequencies.update(caption)

    print(adjective_frequencies.most_common(100))


def check_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nouns",
        help="Path to file containing JSON-serialized list of nouns. ",
        required=True,
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

    noun_stats(parsed_args.nouns, parsed_args.preprocessed_data_folder)
