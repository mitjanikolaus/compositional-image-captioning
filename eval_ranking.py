import argparse
import sys

import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from datasets import *
from tqdm import tqdm
import numpy as np

from train import MODEL_RANKING_GENERATING
from utils import get_splits_from_occurrences_data, BOTTOM_UP_FEATURES_FILENAME

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True  # improve performance if inputs to model are fixed size


def evaluate(data_folder, occurrences_data, checkpoint):
    # Load model
    checkpoint = torch.load(checkpoint, map_location=device)

    model_name = checkpoint["model_name"]
    print("Model: {}".format(model_name))

    encoder = checkpoint["encoder"]
    if encoder:
        encoder = encoder.to(device)
        encoder.eval()

    decoder = checkpoint["decoder"]
    decoder = decoder.to(device)
    decoder.eval()

    print("Decoder params: {}".format(decoder.params))

    indices_non_matching_samples, _, indices_matching_samples = get_splits_from_occurrences_data(
        occurrences_data, 0
    )
    all_indices = indices_non_matching_samples + indices_matching_samples

    if model_name == MODEL_RANKING_GENERATING:
        data_loader = torch.utils.data.DataLoader(
            CaptionTestDataset(data_folder, BOTTOM_UP_FEATURES_FILENAME, all_indices),
            batch_size=1,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
        )
    else:
        raise RuntimeError("Unknown model name: {}".format(model_name))

    # Lists for target captions and generated captions for each image
    embedded_captions = {}
    embedded_images = {}

    for i, (image_features, captions, caption_lengths, coco_id) in enumerate(
        tqdm(data_loader, desc="Embedding samples")
    ):

        image_features = image_features.to(device)
        coco_id = coco_id[0]
        captions = captions[0]
        captions = captions.to(device)
        caption_lengths = caption_lengths.to(device)

        decode_lengths = caption_lengths[0] - 1

        encoded_features = encoder(image_features)

        image_embedded, image_captions_embedded = decoder.forward_ranking(
            encoded_features, captions, decode_lengths
        )

        embedded_images[coco_id] = image_embedded.detach().cpu().numpy()[0]
        embedded_captions[coco_id] = image_captions_embedded.detach().cpu().numpy()

    recall_captions_from_images(
        embedded_images, embedded_captions, indices_matching_samples
    )


def recall_captions_from_images(
    embedded_images, embedded_captions, indices_matching_samples
):
    embedding_size = next(iter(embedded_captions.values())).shape[1]
    all_captions = np.array(list(embedded_captions.values())).reshape(
        -1, embedding_size
    )
    all_captions_keys = list(embedded_captions.keys())

    index_list = []
    ranks = np.zeros(len(indices_matching_samples))
    top1 = np.zeros(len(indices_matching_samples))
    for i, key in enumerate(indices_matching_samples):
        image = embedded_images[key]

        # Compute similarity of image to all captions
        d = np.dot(image, all_captions.T).flatten()
        inds = np.argsort(d)[::-1]
        index_list.append(inds[0])

        # Look for rank of all 5 corresponding captions
        best_rank = len(all_captions)
        index = all_captions_keys.index(key)
        for j in range(5 * index, 5 * index + 5, 1):
            rank = np.where(inds == j)[0]
            if rank < best_rank:
                best_rank = rank
        ranks[i] = best_rank
        top1[i] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1

    print("R@1: {}".format(r1))
    print("R@5: {}".format(r5))
    print("R@10: {}".format(r10))
    print("Median Rank: {}".format(medr))


def check_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-folder",
        help="Folder where the preprocessed data is located",
        default=os.path.expanduser("../datasets/coco2014_preprocessed/"),
    )
    parser.add_argument(
        "--occurrences-data",
        help="File containing occurrences statistics about adjective-noun or verb-noun pairs",
        required=True,
    )
    parser.add_argument(
        "--checkpoint", help="Path to checkpoint of trained model", required=True
    )

    parsed_args = parser.parse_args(args)
    print(parsed_args)
    return parsed_args


if __name__ == "__main__":
    parsed_args = check_args(sys.argv[1:])
    evaluate(
        data_folder=parsed_args.data_folder,
        occurrences_data=parsed_args.occurrences_data,
        checkpoint=parsed_args.checkpoint,
    )
