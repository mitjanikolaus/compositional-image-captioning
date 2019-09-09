"""Analysis of object statistics in sentences with transitive vs. intransitive verbs"""

import argparse
import json
import os
import pickle
import sys

from tqdm import tqdm

import numpy as np

from utils import (
    POS_TAGGED_CAPTIONS_FILENAME,
    NOUNS,
    VERBS,
    PAIR_OCCURENCES,
    OCCURRENCE_DATA,
    get_verbs_for_noun,
    get_objects_for_noun,
    get_objects_for_verb,
)


def noun_stats(preprocessed_data_folder):
    with open(
        os.path.join(preprocessed_data_folder, POS_TAGGED_CAPTIONS_FILENAME), "rb"
    ) as pickle_file:
        captions = pickle.load(pickle_file)

    pairs_with_transitive_verbs = ["eat_horse", "hold_child", "ride_woman", "eat_man"]
    pairs_with_intransitive_verbs = [
        "stand_child",
        "stand_bird",
        "fly_bird",
        "lie_woman",
    ]

    for pair in pairs_with_transitive_verbs + pairs_with_intransitive_verbs:
        print(pair)
        captions_with_objects = 0
        captions_without_objects = 0
        occurrences_data_file = os.path.join(
            "captioning-models", "data", "occurrences", pair + ".json"
        )
        occurrences_data = json.load(open(occurrences_data_file, "r"))
        nouns = occurrences_data[NOUNS]
        verbs = occurrences_data[VERBS]

        matching_images_ids = {
            key
            for key, value in occurrences_data[OCCURRENCE_DATA].items()
            if value[PAIR_OCCURENCES] >= 1
        }

        for coco_id in tqdm(matching_images_ids):
            tagged_captions = captions[coco_id]
            for caption in tagged_captions["pos_tagged_captions"]:
                noun_is_present = False
                verb_is_present = False
                for word in caption.words:
                    if word.lemma in nouns:
                        noun_is_present = True
                    caption_verbs = get_verbs_for_noun(caption, nouns)
                    if caption_verbs & set(verbs):
                        verb_is_present = True
                if noun_is_present and verb_is_present:
                    objects = get_objects_for_noun(
                        caption, nouns
                    ) | get_objects_for_verb(caption, verbs)
                    if len(objects) > 0:
                        captions_with_objects += 1
                    else:
                        captions_without_objects += 1

        print(
            "Captions with objects:",
            np.round(
                captions_with_objects
                / (captions_with_objects + captions_without_objects),
                2,
            ),
        )
        print(
            "Captions without objects:",
            np.round(
                captions_without_objects
                / (captions_with_objects + captions_without_objects),
                2,
            ),
        )
        print("\n")


def check_args(args):
    parser = argparse.ArgumentParser()
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

    noun_stats(parsed_args.preprocessed_data_folder)
