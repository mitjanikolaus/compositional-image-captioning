"""Analysis of the correlation between bounding box sizes to the sizes used in the image captions"""

import argparse
import os
import string
import sys
import numpy as np

import matplotlib.pyplot as plt
import nltk
from nltk import word_tokenize
from pycocotools.coco import COCO
from tqdm import tqdm

nltk.download("punkt", quiet=True)


def analyze_sizes(dataset_folder,):
    data_bbox_sizes = []
    data_described_sizes = []

    for coco_split in ["train2014", "val2014"]:
        ann_file_instances = "{}/annotations/instances_{}.json".format(
            dataset_folder, coco_split
        )
        ann_file_captions = "{}/annotations/captions_{}.json".format(
            dataset_folder, coco_split
        )

        coco_instances = COCO(ann_file_instances)
        coco_captions = COCO(ann_file_captions)

        noun = "cat"
        target_category_id = coco_instances.getCatIds(noun)[0]

        images = coco_instances.loadImgs(coco_instances.getImgIds())

        for img in tqdm(images):
            anns = coco_instances.loadAnns(coco_instances.getAnnIds(imgIds=[img["id"]]))
            biggest_bbox_size = 0
            for ann in anns:
                category_id = ann["category_id"]
                if category_id == target_category_id:
                    if ann["area"] > biggest_bbox_size:
                        biggest_bbox_size = ann["area"]

            if biggest_bbox_size > 0:
                data_bbox_sizes.append(biggest_bbox_size)

                captions = coco_captions.loadAnns(
                    coco_captions.getAnnIds(imgIds=[img["id"]])
                )

                object_described_as_small = False
                object_described_as_big = False
                for caption in captions:
                    caption = caption["caption"].lower()

                    # Remove special chars and punctuation
                    caption = caption.replace("\n", "").replace('"', "")
                    caption = caption.translate(
                        str.maketrans(dict.fromkeys(string.punctuation))
                    )

                    # Tokenize the caption
                    caption = word_tokenize(caption)

                    if {
                        "small",
                        "little",
                        "narrow",
                        "short",
                        "tinier",
                        "tiny",
                        "thin",
                        "compact",
                        "mini",
                        "petite",
                        "skinny",
                    } & set(caption):
                        object_described_as_small = True
                    if {
                        "big",
                        "large",
                        "tall",
                        "huge",
                        "wide",
                        "great",
                        "broad",
                        "enormous",
                        "expansive",
                        "extensive",
                        "giant",
                        "gigantic",
                        "massive",
                        "vast",
                    } & set(caption):
                        object_described_as_big = True

                if object_described_as_small and not object_described_as_big:
                    data_described_sizes.append(0)
                if not object_described_as_small and not object_described_as_big:
                    data_described_sizes.append(1)
                if object_described_as_big and not object_described_as_small:
                    data_described_sizes.append(2)
                if object_described_as_big and object_described_as_small:
                    data_described_sizes.append(3)

    print(
        np.round(
            np.average(
                [
                    bbox_size
                    for bbox_size, size in zip(data_bbox_sizes, data_described_sizes)
                    if size == 0
                ]
            ),
            1,
        )
    )
    print(
        np.round(
            np.std(
                [
                    bbox_size
                    for bbox_size, size in zip(data_bbox_sizes, data_described_sizes)
                    if size == 0
                ]
            ),
            1,
        )
    )
    print(
        len(
            [
                bbox_size
                for bbox_size, size in zip(data_bbox_sizes, data_described_sizes)
                if size == 0
            ]
        )
    )

    print(
        np.round(
            np.average(
                [
                    bbox_size
                    for bbox_size, size in zip(data_bbox_sizes, data_described_sizes)
                    if size == 2
                ]
            ),
            1,
        )
    )
    print(
        np.round(
            np.std(
                [
                    bbox_size
                    for bbox_size, size in zip(data_bbox_sizes, data_described_sizes)
                    if size == 2
                ]
            ),
            1,
        )
    )
    print(
        len(
            [
                bbox_size
                for bbox_size, size in zip(data_bbox_sizes, data_described_sizes)
                if size == 2
            ]
        )
    )

    plt.scatter(data_bbox_sizes, data_described_sizes)
    plt.show()


def check_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-folder",
        help="Folder where the coco dataset is located",
        default=os.path.expanduser("../datasets/coco2014/"),
    )

    parsed_args = parser.parse_args(args)
    print(parsed_args)
    return parsed_args


if __name__ == "__main__":
    parsed_args = check_args(sys.argv[1:])
    analyze_sizes(parsed_args.dataset_folder)
