"""Caption an arbitrary image using a pretrained model"""

import os
import sys

import torch
import json
import argparse

from train import MODEL_SHOW_ATTEND_TELL
from utils import (
    decode_caption,
    WORD_MAP_FILENAME,
    read_image,
    get_caption_without_special_tokens,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_captions(checkpoint, data_folder, image_path, beam_size, print_beam):
    # Load model
    checkpoint = torch.load(checkpoint, map_location=device)
    model_name = checkpoint["model_name"]
    decoder = checkpoint["decoder"]
    decoder = decoder.to(device)
    decoder.eval()

    if not model_name == MODEL_SHOW_ATTEND_TELL:
        raise NotImplementedError()

    encoder = checkpoint["encoder"]
    encoder = encoder.to(device)
    encoder.eval()

    # Load word map
    word_map_path = os.path.join(data_folder, WORD_MAP_FILENAME)
    with open(word_map_path, "r") as json_file:
        word_map = json.load(json_file)

    # Read image and process
    image_data = read_image(image_path)

    image_data = image_data / 255.0

    image_data = torch.FloatTensor(image_data)
    image_features = image_data.unsqueeze(0)
    image_features = image_features.to(device)
    image_features = encoder(image_features)
    generated_sequences, alphas, beam = decoder.beam_search(
        image_features, beam_size, store_alphas=True, print_beam=print_beam
    )

    for seq in generated_sequences:
        print(
            " ".join(
                decode_caption(
                    get_caption_without_special_tokens(seq, word_map), word_map
                )
            )
        )


def check_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument("--image", help="Path to image", required=True)
    parser.add_argument(
        "--checkpoint", help="Path to checkpoint of trained model", required=True
    )
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
    generate_captions(
        parsed_args.checkpoint,
        parsed_args.data_folder,
        parsed_args.image,
        parsed_args.beam_size,
        parsed_args.print_beam,
    )
