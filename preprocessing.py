import argparse
import json
import os
import string
import sys

from collections import Counter
from shutil import copy

import h5py
from nltk import word_tokenize
from pycocotools.coco import COCO
from tqdm import tqdm

from utils import (
    TOKEN_UNKNOWN,
    TOKEN_START,
    TOKEN_END,
    TOKEN_PADDING,
    read_image,
    WORD_MAP_FILENAME,
    IMAGES_FILENAME,
    CAPTIONS_FILENAME,
    CAPTION_LENGTHS_FILENAME,
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


def preprocess_images_and_captions(
    dataset_folder,
    coco_split,
    output_folder,
    vocabulary_size,
    captions_per_image,
    existing_word_map_path,
):

    annFile = "{}/annotations/captions_{}.json".format(dataset_folder, coco_split)
    coco = COCO(annFile)

    images = coco.loadImgs(coco.getImgIds())

    image_paths = []
    image_captions = []

    image_coco_ids = []

    word_freq = Counter()
    max_caption_len = 0

    for img in images:
        captions = []

        annIds = coco.getAnnIds(imgIds=[img["id"]])
        anns = coco.loadAnns(annIds)
        for ann in anns:
            caption = ann["caption"].lower()

            # Remove special chars and punctuation
            caption = caption.replace("\n", "").replace('"', "")
            caption = caption.translate(
                str.maketrans(dict.fromkeys(string.punctuation))
            )

            # Tokenize the caption
            caption = word_tokenize(caption)

            word_freq.update(caption)
            captions.append(caption)

            if len(caption) > max_caption_len:
                max_caption_len = len(caption)

        path = os.path.join(dataset_folder, coco_split, img["file_name"])

        image_paths.append(path)
        image_captions.append(captions)
        image_coco_ids.append(img["id"])

    if existing_word_map_path:
        print("Loading existing word mapping from {}".format(existing_word_map_path))
        with open(existing_word_map_path, "r") as json_file:
            word_map = json.load(json_file)

        copy(existing_word_map_path, output_folder)

    else:
        # Select the most frequent words
        words = [w for w, c in word_freq.most_common(vocabulary_size)]

        # Create word map
        word_map = create_word_map(words)
        word_map_path = os.path.join(output_folder, WORD_MAP_FILENAME)

        print("Saving new word mapping to {}".format(word_map_path))
        with open(word_map_path, "w") as file:
            json.dump(word_map, file)

    # Create hdf5 file and dataset for the images
    images_dataset_path = os.path.join(output_folder, IMAGES_FILENAME)
    print("Creating image dataset at {}".format(images_dataset_path))
    with h5py.File(images_dataset_path, "a") as h5py_file:
        h5py_file.attrs["captions_per_image"] = captions_per_image
        h5py_file.attrs["max_caption_len"] = max_caption_len

        encoded_captions = {}
        caption_lengths = {}

        for i, (path, coco_id) in enumerate(
            tqdm(zip(image_paths, image_coco_ids), total=len(image_paths))
        ):

            # Discard any additional captions
            captions = image_captions[i][:captions_per_image]

            assert len(captions) == captions_per_image

            # Read image and save it to hdf5 file
            img = read_image(path)
            h5py_file.create_dataset(
                str(coco_id), (3, 256, 256), dtype="uint8", data=img
            )

            encoded_captions_for_image = []
            encoded_caption_lengths_for_image = []
            for j, caption in enumerate(captions):
                # Encode caption
                encoded_caption = encode_caption(caption, word_map, max_caption_len)
                encoded_captions_for_image.append(encoded_caption)

                # extend caption length by 2 for start and end of sentence tokens
                caption_length = len(caption) + 2
                encoded_caption_lengths_for_image.append(caption_length)

            encoded_captions[coco_id] = encoded_captions_for_image
            caption_lengths[coco_id] = encoded_caption_lengths_for_image

        # Sanity check
        assert len(h5py_file.keys()) == len(encoded_captions) == len(caption_lengths)

        # Save encoded captions and their lengths to JSON files
        captions_path = os.path.join(output_folder, CAPTIONS_FILENAME)
        print("Saving encoded captions to {}".format(captions_path))
        with open(captions_path, "w") as json_file:
            json.dump(encoded_captions, json_file)
        caption_lengths_path = os.path.join(output_folder, CAPTION_LENGTHS_FILENAME)
        print("Saving caption lengths to {}".format(caption_lengths_path))
        with open(caption_lengths_path, "w") as json_file:
            json.dump(caption_lengths, json_file)


def check_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-folder",
        help="Folder where the coco dataset is located",
        default=os.path.expanduser("~/datasets/coco2014/"),
    )
    parser.add_argument(
        "--coco-split",
        help="Split of the COCO dataset that should be used ('train2014', 'val2014' or 'test2014')",
        default="train2014",
    )
    parser.add_argument(
        "--output-folder",
        help="Folder in which the preprocessed data should be stored",
        default=os.path.expanduser("~/datasets/coco2014_preprocessed/"),
    )
    parser.add_argument(
        "--vocabulary-size",
        help="Number of words that should be saved in the vocabulary",
        type=int,
        default=10000,
    )
    parser.add_argument(
        "--captions-per-image",
        help="Number of captions per image. Additional captions are discarded.",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--word-map",
        help="Path to an existing word map file that should be used instead of creating a new one",
    )

    parsed_args = parser.parse_args(args)
    print(parsed_args)
    return parsed_args


if __name__ == "__main__":
    parsed_args = check_args(sys.argv[1:])
    preprocess_images_and_captions(
        parsed_args.dataset_folder,
        parsed_args.coco_split,
        parsed_args.output_folder,
        parsed_args.vocabulary_size,
        parsed_args.captions_per_image,
        parsed_args.word_map,
    )
