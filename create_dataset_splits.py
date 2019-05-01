import json
import os

import argparse
import sys


from utils import get_splits


def create_dataset_splits(occurrences_data, karpathy_json):

    heldout_pairs = [os.path.basename(file).split(".")[0] for file in occurrences_data]
    train_images_split, val_images_split, test_images_split = get_splits(
        occurrences_data, karpathy_json
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
        "--occurrences-data",
        nargs="+",
        help="Files containing occurrences statistics about adjective-noun or verb-noun pairs",
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
        occurrences_data=parsed_args.occurrences_data,
        karpathy_json=parsed_args.karpathy_json,
    )
