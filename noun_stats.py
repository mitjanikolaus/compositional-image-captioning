from collections import Counter

import argparse
import json
import os
import pickle
import sys

from tqdm import tqdm

from utils import (
    WORD_MAP_FILENAME,
    POS_TAGGED_CAPTIONS_FILENAME,
    get_adjectives_for_noun,
)


def noun_stats(nouns_files, preprocessed_data_folder):
    data = {}
    ids_no_adj = {}

    for nouns_file in nouns_files:
        with open(nouns_file, "r") as json_file:
            nouns = json.load(json_file)

        word_map_path = os.path.join(preprocessed_data_folder, WORD_MAP_FILENAME)
        with open(word_map_path, "r") as json_file:
            word_map = json.load(json_file)

        with open(
            os.path.join(preprocessed_data_folder, POS_TAGGED_CAPTIONS_FILENAME), "rb"
        ) as pickle_file:
            captions = pickle.load(pickle_file)

        first_noun = nouns[0]

        nouns = {noun for noun in nouns if noun in word_map}

        print("Noun stats for: {}".format(nouns))

        adjective_frequencies = Counter()

        for coco_id, tagged_caption in tqdm(captions.items()):
            for i, caption in enumerate(tagged_caption["pos_tagged_captions"]):
                noun_is_present = False
                for token in caption.tokens:
                    if token.text in nouns:
                        noun_is_present = True
                if noun_is_present:
                    adjectives = get_adjectives_for_noun(caption, nouns)
                    if len(adjectives) == 0:
                        adjective_frequencies["No adjective"] += 1

                        if coco_id in ids_no_adj:
                            ids_no_adj[coco_id].add(i)
                        else:
                            ids_no_adj[coco_id] = {1}

                    adjective_frequencies.update(adjectives)

        print(adjective_frequencies.most_common(100))
        data[first_noun] = adjective_frequencies

    data_path = "noun_stats.json"
    print("\nSaving results to {}".format(data_path))
    with open(data_path, "w") as json_file:
        json.dump(data, json_file)

    data_path = "ids_no_adj.json"
    print("\nSaving results to {}".format(data_path))
    with open(data_path, "w") as json_file:
        json.dump(ids_no_adj, json_file)


def check_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nouns",
        nargs="+",
        help="Path to files containing JSON-serialized list of nouns. ",
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
