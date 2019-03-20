import argparse
import json
import os
import sys

import stanfordnlp
from tqdm import tqdm

from utils import (
    WORD_MAP_FILENAME,
    decode_caption,
    get_caption_without_special_tokens,
    PAIR_OCCURENCES,
    ADJECTIVE_OCCURRENCES,
    NOUN_OCCURRENCES,
    NOUNS,
    ADJECTIVES,
    contains_adjective_noun_pair,
    OCCURRENCE_DATA,
    IMAGES_META_FILENAME,
    DATA_CAPTIONS,
    DATA_COCO_SPLIT,
)

# stanfordnlp.download('en', confirm_if_exists=True)


def count_adjective_noun_pairs(
    nouns_file, adjectives_file, preprocessed_data_folder, coco_split
):
    nlp_pipeline = stanfordnlp.Pipeline()

    with open(nouns_file, "r") as json_file:
        nouns = json.load(json_file)
    with open(adjectives_file, "r") as json_file:
        adjectives = json.load(json_file)

    word_map_path = os.path.join(preprocessed_data_folder, WORD_MAP_FILENAME)
    with open(word_map_path, "r") as json_file:
        word_map = json.load(json_file)

    with open(
        os.path.join(preprocessed_data_folder, IMAGES_META_FILENAME), "r"
    ) as json_file:
        images_meta = json.load(json_file)

    print("Looking for pairs: {} - {}".format(adjectives, nouns))

    first_noun = nouns[0]
    first_adjective = adjectives[0]

    nouns = {noun for noun in nouns if noun in word_map}
    adjectives = {adjective for adjective in adjectives if adjective in word_map}

    data = {}
    data[NOUNS] = list(nouns)
    data[ADJECTIVES] = list(adjectives)

    occurrence_data = {}

    for coco_id, image_meta in tqdm(images_meta.items()):
        if image_meta[DATA_COCO_SPLIT] == coco_split:
            encoded_captions = image_meta[DATA_CAPTIONS]

            decoded_captions = [
                " ".join(
                    decode_caption(
                        get_caption_without_special_tokens(caption, word_map), word_map
                    )
                )
                for caption in encoded_captions
            ]

            occurrence_data[coco_id] = {}
            occurrence_data[coco_id][PAIR_OCCURENCES] = 0
            occurrence_data[coco_id][ADJECTIVE_OCCURRENCES] = 0
            occurrence_data[coco_id][NOUN_OCCURRENCES] = 0

            for caption in decoded_captions:
                noun_is_present, adjective_is_present, combination_is_present = contains_adjective_noun_pair(
                    nlp_pipeline, caption, nouns, adjectives
                )
                if combination_is_present:
                    print(caption)
                    occurrence_data[coco_id][PAIR_OCCURENCES] += 1
                if adjective_is_present:
                    occurrence_data[coco_id][ADJECTIVE_OCCURRENCES] += 1
                if noun_is_present:
                    occurrence_data[coco_id][NOUN_OCCURRENCES] += 1

    data[OCCURRENCE_DATA] = occurrence_data

    data_path = "{}_{}_{}.json".format(first_adjective, first_noun, coco_split)
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


def check_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nouns",
        help="Path to file containing JSON-serialized list of nouns. "
        "The first element needs to be a name of a COCO object type.",
        required=True,
    )
    parser.add_argument(
        "--adjectives",
        help="Path to file containing JSON-serialized list of adjectives",
        required=True,
    )
    parser.add_argument(
        "--preprocessed-data-folder",
        help="Folder where the preprocessed data is located (only the word map file is read)",
        default="../datasets/coco2014_preprocessed/",
    )
    parser.add_argument(
        "--coco-split",
        help="COCO split of which the data should be analyzed",
        choices=["train2014", "val2014"],
        required=True,
    )

    parsed_args = parser.parse_args(args)
    print(parsed_args)
    return parsed_args


if __name__ == "__main__":
    parsed_args = check_args(sys.argv[1:])
    count_adjective_noun_pairs(
        parsed_args.nouns,
        parsed_args.adjectives,
        parsed_args.preprocessed_data_folder,
        parsed_args.coco_split,
    )
