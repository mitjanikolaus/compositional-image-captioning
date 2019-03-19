import os
import sys

import h5py
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage.transform
import argparse

from train import MODEL_SHOW_ATTEND_TELL, MODEL_BOTTOM_UP_TOP_DOWN
from utils import (
    decode_caption,
    WORD_MAP_FILENAME,
    IMAGES_FILENAME,
    invert_normalization,
    show_img,
    get_caption_without_special_tokens,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_and_visualize(checkpoint, data_folder, image_id, beam_size, print_beam):
    # Load model
    checkpoint = torch.load(checkpoint, map_location=device)
    model_name = checkpoint["model_name"]
    if model_name == MODEL_BOTTOM_UP_TOP_DOWN:
        raise NotImplementedError(
            "Input conversion to grayscale not supported for this model."
        )

    decoder = checkpoint["decoder"]
    decoder = decoder.to(device)
    decoder.eval()
    if "encoder" in checkpoint and checkpoint["encoder"]:
        encoder = checkpoint["encoder"]
        encoder = encoder.to(device)
        encoder.eval()
    else:
        encoder = None

    # Load word map
    word_map_path = os.path.join(data_folder, WORD_MAP_FILENAME)
    with open(word_map_path, "r") as json_file:
        word_map = json.load(json_file)

    # Read image and process
    h5py_file = h5py.File(os.path.join(data_folder, IMAGES_FILENAME), "r")
    image_data_rgb = h5py_file[image_id].value

    image_data_rgb = image_data_rgb / 255.0
    image_data_grayscale = to_grayscale(image_data_rgb)

    for image_data in [image_data_rgb, image_data_grayscale]:
        show_img(image_data)
        image_data = torch.FloatTensor(image_data)
        image_features = image_data.unsqueeze(0).to(device)
        if encoder:
            image_features = encoder(image_features)
        generated_sequences, alphas, beam = decoder.beam_search(
            image_features, beam_size, print_beam=print_beam
        )

        for sequence in generated_sequences:
            print(
                decode_caption(
                    get_caption_without_special_tokens(sequence, word_map), word_map
                )
            )


def to_grayscale(image_rgb):
    # Transform image to grayscale
    # (cf. https://pillow.readthedocs.io/en/3.2.x/reference/Image.html#PIL.Image.Image.convert)
    r, g, b = image_rgb[0, :, :], image_rgb[1, :, :], image_rgb[2, :, :]
    image_grayscale = 0.2989 * r + 0.5870 * g + 0.1140 * b
    # Make dimensions match the expected input size again
    img = image_grayscale[np.newaxis, :, :]
    img = np.concatenate([img, img, img], axis=0)
    return img


# img = color.rgb2gray(io.imread('image.png'))


def check_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument("--image", help="COCO ID of the image", required=True)
    parser.add_argument("--checkpoint", help="Path to checkpoint of trained model")
    parser.add_argument(
        "--data-folder",
        help="Folder where the preprocessed data is located",
        default="../datasets/coco2014_preprocessed/",
    )
    parser.add_argument(
        "--beam-size", default=5, type=int, help="beam size for beam search"
    )
    parser.add_argument(
        "--print-beam",
        help="Print the decoding beam for every sample",
        default=False,
        action="store_true",
    )

    parsed_args = parser.parse_args(args)
    print(parsed_args)
    return parsed_args


if __name__ == "__main__":
    parsed_args = check_args(sys.argv[1:])
    generate_and_visualize(
        parsed_args.checkpoint,
        parsed_args.data_folder,
        parsed_args.image,
        parsed_args.beam_size,
        parsed_args.print_beam,
    )
