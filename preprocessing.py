import argparse
import json
import os
import string
import sys

from collections import Counter
from shutil import copy

import h5py
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
    IMAGES_META_FILENAME,
    DATA_COCO_SPLIT,
    DATA_CAPTIONS,
    DATA_CAPTION_LENGTHS,
    POS_TAGS_MAP_FILENAME,
    DATA_CAPTIONS_POS,
)

import stanfordnlp

# stanfordnlp.download('en', confirm_if_exists=True)

UNIVERSAL_POS_TAGS = {
    "ADJ",
    "ADP",
    "ADV",
    "AUX",
    "CCONJ",
    "DET",
    "INTJ",
    "NOUN",
    "NUM",
    "PART",
    "PRON",
    "PROPN",
    "PUNCT",
    "SCONJ",
    "SYM",
    "VERB",
    "X",
}


def create_word_map(words):
    word_map = {w: i + 1 for i, w in enumerate(words)}
    # Mapping for special characters
    word_map[TOKEN_UNKNOWN] = len(word_map) + 1
    word_map[TOKEN_START] = len(word_map) + 1
    word_map[TOKEN_END] = len(word_map) + 1
    word_map[TOKEN_PADDING] = 0

    return word_map


def create_pos_map(pos_tags):
    pos_map = {p: i for i, p in enumerate(pos_tags)}
    return pos_map


def encode_caption(caption, word_map, max_caption_len):
    return (
        [word_map[TOKEN_START]]
        + [word_map.get(word, word_map[TOKEN_UNKNOWN]) for word in caption]
        + [word_map[TOKEN_END]]
        + [word_map[TOKEN_PADDING]] * (max_caption_len - len(caption))
    )


def encode_pos_tags(pos_tags, pos_tag_map):
    return [pos_tag_map.get(pos_tag) for pos_tag in pos_tags]


def preprocess_images_and_captions(
    dataset_folder,
    output_folder,
    vocabulary_size,
    captions_per_image,
    existing_word_map_path,
):
    nlp_pipeline = stanfordnlp.Pipeline()
    image_paths = {}
    image_metas = {}

    for coco_split in ["val2014"]:
        annFile = "{}/annotations/captions_{}.json".format(dataset_folder, coco_split)
        coco = COCO(annFile)

        images = coco.loadImgs(coco.getImgIds())

        word_freq = Counter()
        max_caption_len = 0

        for img in tqdm(images[:10]):
            captions = []
            pos_tags = []

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
                doc = nlp_pipeline(caption)
                sentence = doc.sentences[0]
                tokenized_caption = [token.text for token in sentence.tokens]
                pos_tags_caption = [token.words[0].upos for token in sentence.tokens]

                word_freq.update(tokenized_caption)
                captions.append(tokenized_caption)
                pos_tags.append(pos_tags_caption)

                if len(tokenized_caption) > max_caption_len:
                    max_caption_len = len(tokenized_caption)

            path = os.path.join(dataset_folder, coco_split, img["file_name"])

            coco_id = img["id"]
            image_paths[coco_id] = path
            image_metas[coco_id] = {
                DATA_CAPTIONS: captions,
                DATA_COCO_SPLIT: coco_split,
                DATA_CAPTIONS_POS: pos_tags,
            }

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

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

    # Create POS tags map
    pos_tags_map = create_pos_map(UNIVERSAL_POS_TAGS)
    pos_tags_path = os.path.join(output_folder, POS_TAGS_MAP_FILENAME)

    print("Saving new POS tag mapping to {}".format(pos_tags_path))
    with open(pos_tags_path, "w") as file:
        json.dump(pos_tags_map, file)

    # Create hdf5 file and dataset for the images
    images_dataset_path = os.path.join(output_folder, IMAGES_FILENAME)
    print("Creating image dataset at {}".format(images_dataset_path))
    with h5py.File(images_dataset_path, "a") as h5py_file:
        h5py_file.attrs["captions_per_image"] = captions_per_image
        h5py_file.attrs["max_caption_len"] = max_caption_len

        for coco_id, image_path in tqdm(image_paths.items()):

            # Discard any additional captions
            captions = image_metas[coco_id]["captions"][:captions_per_image]
            pos_tags = image_metas[coco_id]["captions_pos"][:captions_per_image]

            assert len(captions) == captions_per_image

            # Read image and save it to hdf5 file
            img = read_image(image_path)
            h5py_file.create_dataset(
                str(coco_id), (3, 256, 256), dtype="uint8", data=img
            )

            encoded_captions = []
            encoded_caption_lengths = []
            for caption in captions:
                # Encode caption
                encoded_caption = encode_caption(caption, word_map, max_caption_len)
                encoded_captions.append(encoded_caption)

                # extend caption length by 2 for start and end of sentence tokens
                caption_length = len(caption) + 2
                encoded_caption_lengths.append(caption_length)

            encoded_pos_tags = []
            for pos_tags_image in pos_tags:
                # Encode POS tags
                encoded_pos_tags.append(encode_pos_tags(pos_tags_image, pos_tags_map))

            image_metas[coco_id][DATA_CAPTIONS] = encoded_captions
            image_metas[coco_id][DATA_CAPTION_LENGTHS] = encoded_caption_lengths
            image_metas[coco_id][DATA_CAPTIONS_POS] = encoded_pos_tags

        # Sanity check
        assert len(h5py_file.keys()) == len(image_metas)

        # Save meta data to JSON file
        captions_path = os.path.join(output_folder, IMAGES_META_FILENAME)
        print("Saving image meta data to {}".format(captions_path))
        with open(captions_path, "w") as json_file:
            json.dump(image_metas, json_file)


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
        parsed_args.output_folder,
        parsed_args.vocabulary_size,
        parsed_args.captions_per_image,
        parsed_args.word_map,
    )
