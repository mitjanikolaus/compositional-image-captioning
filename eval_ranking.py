import argparse
import sys

import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from datasets import *
from tqdm import tqdm

from metrics import recall_captions_from_images, recall_captions_from_images_pairs
from train import MODEL_RANKING_GENERATING
from utils import (
    get_splits_from_occurrences_data,
    BOTTOM_UP_FEATURES_FILENAME,
    get_splits_from_karpathy_json,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True  # improve performance if inputs to model are fixed size

METRIC_RECALL = "recall"
METRIC_RECALL_PAIRS = "recall_pairs"


def evaluate(data_folder, occurrences_data, karpathy_json, checkpoint, metrics):
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

    if occurrences_data and not karpathy_json:
        indices_non_matching_samples, _, indices_matching_samples = get_splits_from_occurrences_data(
            occurrences_data, 0
        )
        all_indices = indices_non_matching_samples + indices_matching_samples
        test_images_split = all_indices
        # Evaluate the model only on the matching samples
        test_indices = indices_matching_samples
    elif karpathy_json and not occurrences_data:
        _, _, test_images_split = get_splits_from_karpathy_json(karpathy_json)
        test_indices = test_images_split
    elif occurrences_data and karpathy_json:
        return ValueError("Specify either karpathy_json or occurrences_data, not both!")
    else:
        return ValueError("Specify either karpathy_json or occurrences_data!")

    print("Test set size: {}".format(len(test_images_split)))
    print("Evaluating performance for {} samples.".format(len(test_indices)))

    if model_name == MODEL_RANKING_GENERATING:
        data_loader = torch.utils.data.DataLoader(
            CaptionTestDataset(
                data_folder, BOTTOM_UP_FEATURES_FILENAME, test_images_split
            ),
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
    target_captions = {}

    for image_features, captions, caption_lengths, coco_id in tqdm(
        data_loader, desc="Embedding samples"
    ):

        image_features = image_features.to(device)
        coco_id = coco_id[0]
        captions = captions[0]
        captions = captions.to(device)
        caption_lengths = caption_lengths.to(device)

        decode_lengths = caption_lengths[0] - 1

        if encoder:
            image_features = encoder(image_features)

        image_embedded, image_captions_embedded = decoder.forward_ranking(
            image_features, captions, decode_lengths
        )

        embedded_images[coco_id] = image_embedded.detach().cpu().numpy()[0]
        embedded_captions[coco_id] = image_captions_embedded.detach().cpu().numpy()
        target_captions[coco_id] = captions.cpu().numpy()

    for metric in metrics:
        calculate_metric(
            metric,
            embedded_images,
            embedded_captions,
            test_indices,
            target_captions,
            word_map,
            occurrences_data,
        )


def calculate_metric(
    metric_name,
    embedded_images,
    embedded_captions,
    test_indices,
    target_captions,
    word_map,
    occurrences_data_file,
):
    if metric_name == METRIC_RECALL:
        recall_captions_from_images(embedded_images, embedded_captions, test_indices)
    elif metric_name == METRIC_RECALL_PAIRS:
        recall_captions_from_images_pairs(
            embedded_images,
            embedded_captions,
            test_indices,
            target_captions,
            word_map,
            occurrences_data_file,
        )


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
    )
    parser.add_argument(
        "--karpathy-json", help="File containing train/val/test split information"
    )
    parser.add_argument(
        "--checkpoint", help="Path to checkpoint of trained model", required=True
    )
    parser.add_argument(
        "--metrics",
        help="Evaluation metrics",
        nargs="+",
        default=[METRIC_RECALL],
        choices=[METRIC_RECALL, METRIC_RECALL_PAIRS],
    )

    parsed_args = parser.parse_args(args)
    print(parsed_args)
    return parsed_args


if __name__ == "__main__":
    parsed_args = check_args(sys.argv[1:])
    evaluate(
        data_folder=parsed_args.data_folder,
        occurrences_data=parsed_args.occurrences_data,
        karpathy_json=parsed_args.karpathy_json,
        checkpoint=parsed_args.checkpoint,
        metrics=parsed_args.metrics,
    )
