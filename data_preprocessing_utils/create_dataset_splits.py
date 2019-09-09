import json
import os

import argparse
import sys

from utils import get_splits_from_occurrences_data

base_dir = os.path.dirname(os.path.abspath(__file__))


def get_splits_from_karpathy_json(karpathy_json):
    with open(karpathy_json, "r") as json_file:
        images_data = json.load(json_file)["images"]

    train_images_split = [
        str(data["cocoid"]) for data in images_data if data["split"] == "train"
    ]

    val_images_split = [
        str(data["cocoid"]) for data in images_data if data["split"] == "val"
    ]

    test_images_split = [
        str(data["cocoid"]) for data in images_data if data["split"] == "test"
    ]

    return train_images_split, val_images_split, test_images_split


def get_splits(heldout_pairs, karpathy_json):
    if heldout_pairs and not karpathy_json:
        train_images_split, val_images_split, test_images_split = get_splits_from_occurrences_data(
            heldout_pairs
        )
    elif karpathy_json and not heldout_pairs:
        train_images_split, val_images_split, test_images_split = get_splits_from_karpathy_json(
            karpathy_json
        )
    elif heldout_pairs and karpathy_json:
        return ValueError("Specify either karpathy_json or heldout_pairs, not both!")
    else:
        return ValueError("Specify either karpathy_json or heldout_pairs!")

    print("Train set size: {}".format(len(train_images_split)))
    print("Val set size: {}".format(len(val_images_split)))
    print("Test set size: {}".format(len(test_images_split)))
    return train_images_split, val_images_split, test_images_split


def create_dataset_splits(heldout_pairs, karpathy_json):
    train_images_split, val_images_split, test_images_split = get_splits(
        heldout_pairs, karpathy_json
    )
    dataset_splits = {
        "train_images_split": train_images_split,
        "val_images_split": val_images_split,
        "test_images_split": test_images_split,
        "heldout_pairs": heldout_pairs,
    }

    file_name = "dataset_splits.json"
    json.dump(dataset_splits, open(file_name, "w"))


def check_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--heldout-pairs",
        nargs="+",
        help="adjective-noun or verb-noun pairs that should be held out",
    )
    parser.add_argument(
        "--karpathy-json", help="File containing train/val/test split information"
    )

    parsed_args = parser.parse_args(args)
    print(parsed_args)
    return parsed_args


if __name__ == "__main__":
    parsed_args = check_args(sys.argv[1:])
    create_dataset_splits(
        heldout_pairs=parsed_args.heldout_pairs, karpathy_json=parsed_args.karpathy_json
    )
