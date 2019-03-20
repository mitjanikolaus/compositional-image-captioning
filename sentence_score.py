import argparse
import json
import os
import sys

import h5py
import torch
import torch.nn.functional as F

from nltk import word_tokenize
from torchvision import transforms

from utils import (
    TOKEN_START,
    TOKEN_END,
    WORD_MAP_FILENAME,
    IMAGES_FILENAME,
    IMAGENET_IMAGES_MEAN,
    IMAGENET_IMAGES_STD,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sequence_score(encoder, decoder, img, word_map, sequence):
    """Calculate the sequence score for a given image-sequence-pair."""

    # Move image to GPU device, if available
    image = img.to(device)

    # Encode
    encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
    encoder_dim = encoder_out.size(3)

    # Flatten encoding
    encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)

    # Encode sequence
    encoded_sequence = [word_map[token] for token in word_tokenize(sequence)]

    # Add start and end token
    encoded_sequence = (
        [word_map[TOKEN_START]] + encoded_sequence + [word_map[TOKEN_END]]
    )

    # Transform to a tensor
    encoded_sequence = torch.LongTensor(encoded_sequence, device=device)

    score = 0

    # Start decoding
    states = decoder.init_hidden_states(encoder_out)
    for step in range(0, len(encoded_sequence) - 1):
        # Embed the word of the previous timestep
        embeddings = decoder.word_embedding(encoded_sequence[step]).unsqueeze(0)

        # Perform a forward step
        predictions, states, alpha = decoder.forward_step(
            encoder_out, embeddings, states
        )
        scores = F.log_softmax(predictions, dim=1).squeeze(0)

        # Add the score for the next word in the given sequence
        score += scores[encoded_sequence[step + 1]]

    return score


def load_model_and_calculate_score(checkpoint, data_folder, image_id, sequences):
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
    h5py_file = h5py.File(os.path.join(data_folder, IMAGES_FILENAME), "r")
    image_data = h5py_file[image_id].value

    image = torch.FloatTensor(image_data / 255.0)
    normalize = transforms.Normalize(mean=IMAGENET_IMAGES_MEAN, std=IMAGENET_IMAGES_STD)
    image = normalize(image).unsqueeze(0)

    for sequence in sequences:
        score = sequence_score(encoder, decoder, image, word_map, sequence)

        print("{} \tScore: {}".format(sequence, score))


def check_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument("--image", help="COCO ID of the image", required=True)
    parser.add_argument(
        "--checkpoint", help="Path to checkpoint of trained model", required=True
    )
    parser.add_argument(
        "--data-folder",
        help="Folder where the preprocessed data is located",
        default="../datasets/coco2014_preprocessed/",
    )
    parser.add_argument(
        "--sequences",
        help="Sequences of which the score should be calculated",
        required=True,
        nargs="+",
    )

    parsed_args = parser.parse_args(args)
    print(parsed_args)
    return parsed_args


if __name__ == "__main__":
    parsed_args = check_args(sys.argv[1:])
    load_model_and_calculate_score(
        parsed_args.checkpoint,
        parsed_args.data_folder,
        parsed_args.image,
        parsed_args.sequences,
    )
