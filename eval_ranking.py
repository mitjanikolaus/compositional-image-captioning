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
    word_map = decoder.word_map
    decoder.eval()

    print("Decoder params: {}".format(decoder.params))

    indices_non_matching_samples, _, indices_matching_samples = get_splits_from_occurrences_data(
        occurrences_data, 0
    )

    if model_name == MODEL_RANKING_GENERATING:
        data_loader_non_matching = torch.utils.data.DataLoader(
            CaptionTrainDataset(
                data_folder, BOTTOM_UP_FEATURES_FILENAME, indices_non_matching_samples
            ),
            batch_size=50,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
        )
        data_loader_matching = torch.utils.data.DataLoader(
            CaptionTrainDataset(
                data_folder, BOTTOM_UP_FEATURES_FILENAME, indices_matching_samples
            ),
            batch_size=50,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
        )
    else:
        raise RuntimeError("Unknown model name: {}".format(model_name))

    # Lists for target captions and generated captions for each image
    embedded_captions_non_matching = []
    embedded_captions_matching = []
    embedded_images_non_matching = []
    embedded_images_matching = []

    for i, (image_features, captions, caption_lengths, coco_ids) in enumerate(
        tqdm(data_loader_matching, desc="Embedding matching samples")
    ):

        image_features = image_features.to(device)
        captions = captions.to(device)
        caption_lengths = caption_lengths.to(device)

        encoded_features = encoder(image_features)

        images_embedded, captions_embedded = decoder.forward_ranking(
            encoded_features, captions, caption_lengths - 1
        )

        embedded_captions_matching.extend(captions_embedded.detach().numpy())
        embedded_images_matching.extend(images_embedded.detach().numpy())

    for i, (image_features, captions, caption_lengths, coco_ids) in enumerate(
        tqdm(data_loader_non_matching, desc="Embedding non-matching samples")
    ):

        image_features = image_features.to(device)
        captions = captions.to(device)
        caption_lengths = caption_lengths.to(device)

        encoded_features = encoder(image_features)

        images_embedded, captions_embedded = decoder.forward_ranking(
            encoded_features, captions, caption_lengths - 1
        )

        embedded_captions_non_matching.extend(captions_embedded.detach().numpy())
        embedded_images_non_matching.extend(images_embedded.detach().numpy())

    recall_captions_from_images(
        embedded_images_matching,
        embedded_images_non_matching,
        embedded_captions_matching,
        embedded_captions_non_matching,
    )


def order_sim(im, s):
    """Order embeddings similarity measure $max(0, s-im)$
    """
    YmX = s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1)) - im.unsqueeze(
        0
    ).expand(s.size(0), im.size(0), s.size(1))
    score = -YmX.clamp(min=0).pow(2).sum(2).sqrt().t()
    return score


def recall_captions_from_images(
    embedded_images_matching,
    embedded_images_non_matching,
    embedded_captions_matching,
    embedded_captions_non_matching,
    return_ranks=False,
):
    """
    Images->Text (Image Annotation)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    # ranks = np.zeros(npts)
    # top1 = np.zeros(npts)

    all_captions = np.array(embedded_captions_matching + embedded_captions_non_matching)

    index_list = []
    ranks = np.zeros(len(embedded_images_matching))
    top1 = np.zeros(len(embedded_images_matching))
    for index, image in enumerate(embedded_images_matching):
        # Compute scores
        d = np.dot(image, all_captions.T).flatten()
        inds = np.argsort(d)[::-1]
        index_list.append(inds[0])

        # Score
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = np.where(inds == i)[0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    print("R@1: {}".format(r1))
    print("R@5: {}".format(r5))
    print("R@10: {}".format(r10))

    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


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
