import argparse
import os
import sys
import json

import h5py
import nltk
from tqdm import tqdm

from utils import (
    TOKEN_UNKNOWN,
    TOKEN_START,
    TOKEN_END,
    TOKEN_PADDING,
    read_image,
    TEST_IMAGES_FILENAME,
)


def create_word_map(words):
    word_map = {w: i + 1 for i, w in enumerate(words)}
    # Mapping for special characters
    word_map[TOKEN_UNKNOWN] = len(word_map) + 1
    word_map[TOKEN_START] = len(word_map) + 1
    word_map[TOKEN_END] = len(word_map) + 1
    word_map[TOKEN_PADDING] = 0

    return word_map


def encode_caption(caption, word_map, max_caption_len):
    return (
        [word_map[TOKEN_START]]
        + [word_map.get(word, word_map[TOKEN_UNKNOWN]) for word in caption]
        + [word_map[TOKEN_END]]
        + [word_map[TOKEN_PADDING]] * (max_caption_len - len(caption))
    )


def preprocess_images(dataset_folder, output_folder, captions_per_image):
    image_paths = {}

    for root, dirnames, filenames in os.walk(os.path.join(dataset_folder, "test2015")):
        for filename in filenames:
            coco_id = int(str(filename.split("_")[-1]).split(".")[0])

            path = os.path.join(dataset_folder, "test2015", filename)

            image_paths[coco_id] = path

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Create hdf5 file and dataset for the images
    images_dataset_path = os.path.join(output_folder, TEST_IMAGES_FILENAME)
    print("Creating image dataset at {}".format(images_dataset_path))
    with h5py.File(images_dataset_path, "a") as h5py_file:
        h5py_file.attrs["captions_per_image"] = captions_per_image

        for coco_id, image_path in tqdm(image_paths.items()):

            # Read image and save it to hdf5 file
            img = read_image(image_path)
            h5py_file.create_dataset(
                str(coco_id), (3, 256, 256), dtype="uint8", data=img
            )

    coco_ids = list(image_paths.keys())
    json.dump(coco_ids, open("coco_test_ids.json", "w"))


def check_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-folder",
        help="Folder where the coco dataset is located",
        default=os.path.expanduser("../datasets/coco2014/"),
    )
    parser.add_argument(
        "--output-folder",
        help="Folder in which the preprocessed data should be stored",
        default=os.path.expanduser("../datasets/coco2014_preprocessed/"),
    )
    parser.add_argument(
        "--captions-per-image",
        help="Number of captions per image. Additional captions are discarded.",
        type=int,
        default=5,
    )

    parsed_args = parser.parse_args(args)
    print(parsed_args)
    return parsed_args


if __name__ == "__main__":
    parsed_args = check_args(sys.argv[1:])
    preprocess_images(
        parsed_args.dataset_folder,
        parsed_args.output_folder,
        parsed_args.captions_per_image,
    )
