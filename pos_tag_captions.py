import argparse
import json
import os
import pickle
import sys

import stanfordnlp
from tqdm import tqdm

from utils import (
    WORD_MAP_FILENAME,
    decode_caption,
    get_caption_without_special_tokens,
    IMAGES_META_FILENAME,
    DATA_CAPTIONS,
    DATA_COCO_SPLIT,
    POS_TAGGED_CAPTIONS_FILENAME,
)

# stanfordnlp.download('en', confirm_if_exists=True)


def count_adjective_noun_pairs(preprocessed_data_folder):
    nlp_pipeline = stanfordnlp.Pipeline()

    with open(
        os.path.join(preprocessed_data_folder, IMAGES_META_FILENAME), "r"
    ) as json_file:
        images_meta = json.load(json_file)

    word_map_path = os.path.join(preprocessed_data_folder, WORD_MAP_FILENAME)
    with open(word_map_path, "r") as json_file:
        word_map = json.load(json_file)

    data = {}

    for coco_id, image_meta in tqdm(images_meta.items()):
        encoded_captions = image_meta[DATA_CAPTIONS]

        decoded_captions = [
            " ".join(
                decode_caption(
                    get_caption_without_special_tokens(caption, word_map), word_map
                )
            )
            for caption in encoded_captions
        ]

        data[coco_id] = {}
        data[coco_id][DATA_COCO_SPLIT] = image_meta[DATA_COCO_SPLIT]
        data[coco_id]["pos_tagged_captions"] = []

        for caption in decoded_captions:
            doc = nlp_pipeline(caption)
            sentence = doc.sentences[0]
            data[coco_id]["pos_tagged_captions"].append(sentence)

    data_path = os.path.join(preprocessed_data_folder, POS_TAGGED_CAPTIONS_FILENAME)
    print("\nSaving results to {}".format(data_path))
    with open(data_path, "wb") as pickle_file:
        pickle.dump(data, pickle_file)


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
    count_adjective_noun_pairs(parsed_args.preprocessed_data_folder)
