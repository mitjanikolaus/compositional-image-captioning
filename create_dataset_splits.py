import json
import os

import argparse
import sys

from utils import get_splits

base_dir = os.path.dirname(os.path.abspath(__file__))


def create_dataset_splits(heldout_pairs, karpathy_json):
    occurrences_data_files = [
        os.path.join(base_dir, "data", "occurrences", pair + ".json")
        for pair in heldout_pairs
    ]
    train_images_split, val_images_split, test_images_split = get_splits(
        occurrences_data_files, karpathy_json
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
