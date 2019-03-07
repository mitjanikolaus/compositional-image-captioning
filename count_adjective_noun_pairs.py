import argparse
import json
import os
import sys

import stanfordnlp
from pycocotools.coco import COCO
import nltk
from tqdm import tqdm

from utils import (
    WORD_MAP_FILENAME,
    CAPTIONS_FILENAME,
    decode_caption,
    get_caption_without_special_tokens,
)

nltk.download("wordnet", quiet=True)

# stanfordnlp.download('en', confirm_if_exists=True)

RELATION_NOMINAL_SUBJECT = "nsubj"
RELATION_ADJECTIVAL_MODIFIER = "amod"
RELATION_CONJUNCT = "conj"


def contains_adjective_noun_pair(nlp_pipeline, caption, nouns, adjectives):
    noun_is_present = False
    adjective_is_present = False

    doc = nlp_pipeline(caption)
    sentence = doc.sentences[0]

    for token in sentence.tokens:
        if token.text in nouns:
            noun_is_present = True
        if token.text in adjectives:
            adjective_is_present = True

    dependencies = sentence.dependencies
    caption_adjectives = {
        d[2].text
        for d in dependencies
        if d[1] == RELATION_ADJECTIVAL_MODIFIER and d[0].text in nouns
    } | {
        d[0].text
        for d in dependencies
        if d[1] == RELATION_NOMINAL_SUBJECT and d[2].text in nouns
    }
    conjuncted_caption_adjectives = set()
    for adjective in caption_adjectives:
        conjuncted_caption_adjectives.update(
            {
                d[2].text
                for d in dependencies
                if d[1] == RELATION_CONJUNCT and d[0].text == adjective
            }
            | {
                d[2].text
                for d in dependencies
                if d[1] == RELATION_ADJECTIVAL_MODIFIER and d[0].text == adjective
            }
        )

    caption_adjectives |= conjuncted_caption_adjectives
    combination_is_present = bool(adjectives & caption_adjectives)

    return noun_is_present, adjective_is_present, combination_is_present


def count_adjective_noun_pairs(
    nouns_file, adjectives_file, preprocessed_data_folder, dataset_folder
):
    nlp_pipeline = stanfordnlp.Pipeline()

    dataType = "train2014"

    annFile = "{}/annotations/instances_{}.json".format(dataset_folder, dataType)
    coco = COCO(annFile)

    with open(nouns_file, "r") as json_file:
        nouns = json.load(json_file)
    with open(adjectives_file, "r") as json_file:
        adjectives = json.load(json_file)

    word_map_path = os.path.join(preprocessed_data_folder, WORD_MAP_FILENAME)
    with open(word_map_path, "r") as json_file:
        word_map = json.load(json_file)

    first_noun = nouns[0]
    first_adjective = adjectives[0]
    catIds = coco.getCatIds(catNms=[first_noun])
    imgIds = coco.getImgIds(catIds=catIds)

    print("Found {} {} images.".format(len(imgIds), nouns[0]))

    print("Looking for pairs: {} - {}".format(adjectives, nouns))

    nouns = {noun for noun in nouns if noun in word_map}
    adjectives = {adjective for adjective in adjectives if adjective in word_map}

    # Load captions
    with open(
        os.path.join(preprocessed_data_folder, CAPTIONS_FILENAME), "r"
    ) as json_file:
        all_captions = json.load(json_file)

    data = {}

    for i, coco_id in enumerate(tqdm(imgIds)):
        encoded_captions = all_captions[str(coco_id)]

        # TODO is join with spaces good enough?
        decoded_captions = [
            " ".join(
                decode_caption(
                    get_caption_without_special_tokens(caption, word_map), word_map
                )
            )
            for caption in encoded_captions
        ]

        data[coco_id] = {}
        data[coco_id]["pair_occurrences"] = 0
        data[coco_id]["adjective_occurrences"] = 0
        data[coco_id]["noun_occurrences"] = 0

        for caption in decoded_captions:
            noun_is_present, adjective_is_present, combination_is_present = contains_adjective_noun_pair(
                nlp_pipeline, caption, nouns, adjectives
            )
            if combination_is_present:
                print(caption)
                data[coco_id]["pair_occurrences"] += 1
            if adjective_is_present:
                data[coco_id]["adjective_occurrences"] += 1
            if noun_is_present:
                data[coco_id]["noun_occurrences"] += 1

    data_path = "{}_{}.json".format(first_adjective, first_noun)
    print("\nSaving results to {}".format(data_path))
    with open(data_path, "w") as json_file:
        json.dump(data, json_file)

    for n in range(1, 5):
        noun_occurences = len(
            [
                image_data
                for image_data in data.values()
                if image_data["noun_occurrences"] >= n
            ]
        )
        adjective_occurences = len(
            [
                image_data
                for image_data in data.values()
                if image_data["adjective_occurrences"] >= n
            ]
        )
        pair_occurences = len(
            [
                image_data
                for image_data in data.values()
                if image_data["pair_occurrences"] >= n
            ]
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
        "-N",
        "--nouns",
        help="Path to file containing JSON-serialized list of nouns. "
        "The first element needs to be a name of a COCO object type.",
        required=True,
    )
    parser.add_argument(
        "-A",
        "--adjectives",
        help="Path to file containing JSON-serialized list of adjectives",
        required=True,
    )
    parser.add_argument(
        "--preprocessed-data-folder",
        help="Folder where the preprocessed data is located (only the word map file is read)",
        default=os.path.expanduser("~/datasets/coco2014_preprocessed/"),
    )
    parser.add_argument(
        "--dataset-folder",
        help="Folder where the coco dataset is located (only the annotation file is read)",
        default=os.path.expanduser("~/datasets/coco2014/"),
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
        parsed_args.dataset_folder,
    )
