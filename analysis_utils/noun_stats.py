"""Calculate some statistics for the nouns in the descriptions in the COCO dataset"""

from collections import Counter

import argparse
import json
import os
import pickle
import sys

from tqdm import tqdm

from utils import (
    POS_TAGGED_CAPTIONS_FILENAME,
    get_adjectives_for_noun,
    get_verbs_for_noun,
    get_objects_for_noun,
)


ADJECTIVES_COLORS = {
    "white",
    "black",
    "red",
    "green",
    "blue",
    "yellow",
    "brown",
    "gray",
    "purple",
    "dark",
    "grey",
    "pink",
    "silver",
    "golden",
    "orange",
    "beige",
    "neon",
    "gold",
    "black-and-white",
    "bronze",
    "blonde",
    "violet",
    "maroon",
    "turquoise",
}
ADJECTIVES_SIZES = {
    "big",
    "broad",
    "compact",
    "enormous",
    "expansive",
    "extensive",
    "giant",
    "gigantic",
    "great",
    "huge",
    "large",
    "little",
    "long",
    "massive",
    "mini",
    "narrow",
    "petite",
    "short",
    "sized",
    "skinny",
    "small",
    "tall",
    "thin",
    "tiny",
    "vast",
    "wide",
}

ADJECTIVES_AGES = {
    "adolescent",
    "aged",
    "ancient",
    "decrepit",
    "elder",
    "elderly",
    "grown",
    "immature",
    "infant",
    "junior",
    "juvenile",
    "middle-aged",
    "old",
    "older",
    "retired",
    "retro",
    "senior",
    "teen",
    "teenage",
    "teenaged",
    "vintage",
    "worn",
    "young",
    "youth",
}

GROUP_COLORS = "colors"
GROUP_SIZES = "sizes"
GROUP_AGES = "ages"

GROUP_OTHERS = "others"


def get_adjective_group(adjective):
    if adjective in ADJECTIVES_COLORS:
        return GROUP_COLORS
    elif adjective in ADJECTIVES_SIZES:
        return GROUP_SIZES
    elif adjective in ADJECTIVES_AGES:
        return GROUP_AGES
    else:
        return GROUP_OTHERS


def noun_stats(nouns_files, preprocessed_data_folder):
    data = {}

    for nouns_file in nouns_files:
        with open(nouns_file, "r") as json_file:
            nouns = json.load(json_file)

        with open(
            os.path.join(preprocessed_data_folder, POS_TAGGED_CAPTIONS_FILENAME), "rb"
        ) as pickle_file:
            captions = pickle.load(pickle_file)

        first_noun = nouns[0]

        print("Noun stats for: {}".format(nouns))

        total = 0
        adjective_frequencies = Counter()
        verb_frequencies = Counter()
        adjective_group_frequencies = Counter()

        object_counts = Counter()

        for coco_id, tagged_caption in tqdm(captions.items()):
            for caption in tagged_caption["pos_tagged_captions"]:
                noun_is_present = False
                for word in caption.words:
                    if word.lemma in nouns:
                        noun_is_present = True
                if noun_is_present:
                    adjectives = get_adjectives_for_noun(caption, nouns)
                    if len(adjectives) == 0:
                        adjective_frequencies["No adjective"] += 1
                    adjective_frequencies.update(adjectives)

                    verbs = get_verbs_for_noun(caption, nouns)
                    if len(verbs) == 0:
                        verb_frequencies["No verb"] += 1
                    verb_frequencies.update(verbs)

                    adjective_group_frequencies.update(
                        {get_adjective_group(adjective) for adjective in adjectives}
                    )

                    objects = get_objects_for_noun(caption, nouns)
                    object_counts.update([len(objects)])

                    total += 1

        print("Total: ", total)
        print(adjective_frequencies.most_common(20))
        print(verb_frequencies.most_common(20))
        print(adjective_group_frequencies.most_common(20))
        print(object_counts.most_common(20))

        data[first_noun] = {}
        data[first_noun]["total"] = total
        data[first_noun]["adjective_frequencies"] = adjective_frequencies
        data[first_noun]["verb_frequencies"] = verb_frequencies
        data[first_noun]["adjective_group_frequencies"] = verb_frequencies
        data[first_noun]["object_counts"] = object_counts

    data_path = "noun_stats.json"
    print("\nSaving results to {}".format(data_path))
    with open(data_path, "w") as json_file:
        json.dump(data, json_file)


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
