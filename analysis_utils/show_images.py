"""Display images for the given COCO IDs"""

import os
import sys

import h5py
import argparse

from utils import IMAGES_FILENAME, show_img


def show_images(data_folder, image_ids):
    # Read image and process
    h5py_file = h5py.File(os.path.join(data_folder, IMAGES_FILENAME), "r")
    for image_id in image_ids:
        image_data = h5py_file[image_id].value
        show_img(image_data)


def check_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument("--images", help="COCO IDs of the images to show", nargs="+")
    parser.add_argument(
        "--data-folder",
        help="Folder where the preprocessed data is located",
        default="../datasets/coco2014_preprocessed/",
    )

    parsed_args = parser.parse_args(args)
    print(parsed_args)
    return parsed_args


if __name__ == "__main__":
    parsed_args = check_args(sys.argv[1:])
    show_images(parsed_args.data_folder, parsed_args.images)
