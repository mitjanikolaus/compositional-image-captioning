import os
import sys

import torch
import numpy as np
import json
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage.transform
import argparse
from PIL import Image

from inference import generate_captions
from utils import (
    decode_caption,
    read_image,
    IMAGENET_IMAGES_MEAN,
    IMAGENET_IMAGES_STD,
    WORD_MAP_FILENAME,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def visualize_attention(image_path, encoded_caption, alphas, word_map, smooth=True):
    """
    Visualizes caption with weights at every word.

    Adapted from paper authors' repo: https://github.com/kelvinxu/arctic-captions/blob/master/alpha_visualization.ipynb

    """
    image = Image.open(image_path)
    image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)

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
        plt.imshow(image)
        current_alpha = alphas[t]
        if smooth:
            alpha = skimage.transform.pyramid_expand(current_alpha, upscale=24, sigma=8)
        else:
            alpha = skimage.transform.resize(current_alpha, [14 * 24, 14 * 24])
        if t == 0:
            plt.imshow(alpha, alpha=0)
        else:
            plt.imshow(alpha, alpha=0.8)
        plt.set_cmap(cm.Greys_r)
        plt.axis("off")
    plt.show()


def generate_and_visualize(checkpoint, data_folder, img_path, beam_size, smoothen):
    # Load model
    checkpoint = torch.load(checkpoint, map_location=device)
    decoder = checkpoint["decoder"]
    decoder = decoder.to(device)
    decoder.eval()
    encoder = checkpoint["encoder"]
    encoder = encoder.to(device)
    encoder.eval()

    # Load word map
    word_map_path = os.path.join(data_folder, WORD_MAP_FILENAME)
    with open(word_map_path, "r") as json_file:
        word_map = json.load(json_file)

    # Read image and process
    image = read_image(img_path)
    image = torch.FloatTensor(image / 255.0)
    normalize = transforms.Normalize(mean=IMAGENET_IMAGES_MEAN, std=IMAGENET_IMAGES_STD)
    transform = transforms.Compose([normalize])
    image = transform(image)
    image = image.unsqueeze(0)

    seq, alphas = generate_captions(
        encoder, decoder, image, word_map, beam_size, store_alphas=True
    )

    # Visualize caption and attention of best sequence
    visualize_attention(img_path, seq[0], np.array(alphas[0]), word_map, smoothen)


def check_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument("--image", help="Path to the image", required=True)
    parser.add_argument(
        "--checkpoint",
        help="Path to checkpoint of trained model",
        default="best_checkpoint.pth.tar",
    )
    parser.add_argument(
        "--data-folder",
        help="Folder where the preprocessed data is located (only the word map file is read)",
        default=os.path.expanduser("~/datasets/coco2014_preprocessed/"),
    )
    parser.add_argument(
        "--beam-size", default=5, type=int, help="beam size for beam search"
    )
    parser.add_argument(
        "--dont_smoothen",
        dest="smoothen",
        action="store_false",
        help="Do not smoothen alpha overlay",
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
        parsed_args.smoothen,
    )
