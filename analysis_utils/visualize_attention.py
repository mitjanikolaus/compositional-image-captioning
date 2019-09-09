"""Visualize the attention heatmap for the show, attend and tell model"""
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

from utils import (
    decode_caption,
    WORD_MAP_FILENAME,
    IMAGES_FILENAME,
    invert_normalization,
    MODEL_SHOW_ATTEND_TELL,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def visualize_attention(image, encoded_caption, alphas, word_map, smoothen):
    """
    Visualizes caption with weights at every word.

    Adapted from paper authors' repo: https://github.com/kelvinxu/arctic-captions/blob/master/alpha_visualization.ipynb

    """
    if torch.min(image).data < 0:
        image = invert_normalization(image)

    decoded_caption = decode_caption(encoded_caption, word_map)

    for t in range(len(decoded_caption)):
        if t > 50:
            break
        plt.subplot(np.ceil(len(decoded_caption) / 5.0), 5, t + 1)

        plt.text(
            0,
            1,
            "%s" % (decoded_caption[t]),
            color="black",
            backgroundcolor="white",
            fontsize=12,
        )
        plt.imshow(image.numpy().transpose(1, 2, 0))
        alphas = np.array(alphas)
        if smoothen:
            alpha = skimage.transform.pyramid_expand(
                alphas[t],
                upscale=(image.shape[1] / alphas[t].shape[0]),
                multichannel=False,
            )
        else:
            alpha = skimage.transform.resize(
                alphas[t], [image.shape[1], image.shape[1]]
            )
        if t == 0:
            plt.imshow(alpha, alpha=0)
        else:
            plt.imshow(alpha, alpha=0.8)
        plt.set_cmap(cm.Greys_r)
        plt.axis("off")
    plt.show()


def generate_and_visualize(
    checkpoint, data_folder, image_id, beam_size, print_beam, smoothen
):
    # Load model
    checkpoint = torch.load(checkpoint, map_location=device)
    model_name = checkpoint["model_name"]
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
    image_data = h5py_file[image_id].value

    if model_name == MODEL_SHOW_ATTEND_TELL:
        image_data = image_data / 255.0

    image_data = torch.FloatTensor(image_data)
    image_features = image_data.unsqueeze(0)
    image_features = image_features.to(device)
    if encoder:
        image_features = encoder(image_features)
    generated_sequences, alphas, beam = decoder.beam_search(
        image_features, beam_size, store_alphas=True, print_beam=print_beam
    )

    # Visualize caption and attention of best sequence
    visualize_attention(
        image_data, generated_sequences[0], alphas[0], word_map, smoothen
    )


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
    parser.add_argument(
        "--smoothen",
        dest="smoothen",
        action="store_true",
        help="Smoothen the alpha overlay",
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
        parsed_args.smoothen,
    )
