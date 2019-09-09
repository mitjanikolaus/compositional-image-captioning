"""Display images that match the given concept pair"""

import sys

import argparse

import h5py
import json
import os

from utils import (
    IMAGES_META_FILENAME,
    DATA_CAPTIONS,
    IMAGES_FILENAME,
    show_img,
    WORD_MAP_FILENAME,
    decode_caption,
    get_caption_without_special_tokens,
    get_splits_from_occurrences_data,
)


def show_images(data_folder, pair):
    image_features = h5py.File(os.path.join(data_folder, IMAGES_FILENAME), "r")

    with open(os.path.join(data_folder, IMAGES_META_FILENAME), "r") as json_file:
        images_meta = json.load(json_file)

    word_map_file = os.path.join(data_folder, WORD_MAP_FILENAME)
    with open(word_map_file, "r") as json_file:
        word_map = json.load(json_file)

    _, _, test_images_split = get_splits_from_occurrences_data([pair])

    for coco_id in test_images_split:
        image_data = image_features[coco_id][()]

        print("COCO ID: ", coco_id)
        for caption in images_meta[coco_id][DATA_CAPTIONS]:
            print(
                " ".join(
                    decode_caption(
                        get_caption_without_special_tokens(caption, word_map), word_map
                    )
                )
            )
        show_img(image_data)
        print("")


def check_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-folder",
        help="Folder where the preprocessed data is located",
        default=os.path.expanduser("../datasets/coco2014_preprocessed/"),
    )
    parser.add_argument(
        "--pair", help="adjective-noun or verb-noun pair", required=True
    )

    parsed_args = parser.parse_args(args)
    print(parsed_args)
    return parsed_args


if __name__ == "__main__":
    parsed_args = check_args(sys.argv[1:])
    show_images(parsed_args.data_folder, parsed_args.pair)
