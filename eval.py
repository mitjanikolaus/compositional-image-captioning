import argparse
import sys

import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from datasets import *
from metrics import recall_adjective_noun_pairs
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm

from train import MODEL_SHOW_ATTEND_TELL, MODEL_BOTTOM_UP_TOP_DOWN
from utils import (
    get_caption_without_special_tokens,
    IMAGENET_IMAGES_MEAN,
    WORD_MAP_FILENAME,
    IMAGENET_IMAGES_STD,
    get_splits_from_occurrences_data,
    IMAGES_FILENAME,
    BOTTOM_UP_FEATURES_FILENAME,
)
from visualize_attention import visualize_attention

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True  # improve performance if inputs to model are fixed size


def evaluate(
    data_folder,
    occurrences_data,
    checkpoint,
    metrics,
    beam_size,
    max_caption_len,
    visualize,
    print_beam,
):
    # Load model
    checkpoint = torch.load(checkpoint, map_location=device)
    decoder = checkpoint["decoder"]
    decoder = decoder.to(device)
    decoder.eval()

    encoder = checkpoint["encoder"]
    if encoder:
        encoder = encoder.to(device)
        encoder.eval()

    model_name = checkpoint["model_name"]

    # Load word map
    word_map_path = os.path.join(data_folder, WORD_MAP_FILENAME)
    with open(word_map_path, "r") as json_file:
        word_map = json.load(json_file)

    _, _, test_images_split = get_splits_from_occurrences_data(occurrences_data)

    if model_name == MODEL_SHOW_ATTEND_TELL:
        # Normalization
        normalize = transforms.Normalize(
            mean=IMAGENET_IMAGES_MEAN, std=IMAGENET_IMAGES_STD
        )

        # DataLoader
        data_loader = torch.utils.data.DataLoader(
            CaptionTestDataset(
                data_folder,
                IMAGES_FILENAME,
                test_images_split,
                transforms.Compose([normalize]),
                features_scale_factor=1 / 255.0,
            ),
            batch_size=1,
            shuffle=True,
            num_workers=1,
            pin_memory=True,
        )
    elif model_name == MODEL_BOTTOM_UP_TOP_DOWN:
        data_loader = torch.utils.data.DataLoader(
            CaptionTestDataset(
                data_folder, BOTTOM_UP_FEATURES_FILENAME, test_images_split
            ),
            batch_size=1,
            shuffle=True,
            num_workers=1,
            pin_memory=True,
        )
    else:
        raise RuntimeError("Unknown model name: {}".format(model_name))

    # Lists for target captions and generated captions for each image
    target_captions = []
    generated_captions = []
    coco_ids = []

    for i, (image_features, all_captions_for_image, coco_id) in enumerate(
        tqdm(data_loader, desc="Evaluate with beam size " + str(beam_size))
    ):

        # Target captions
        target_captions.append(
            [
                get_caption_without_special_tokens(caption, word_map)
                for caption in all_captions_for_image[0].tolist()
            ]
        )

        # Generate captions
        encoded_features = image_features.to(device)
        if encoder:
            encoded_features = encoder(image_features)

        if visualize:
            top_k_generated_captions, alphas = decoder.beam_search(
                encoded_features,
                beam_size,
                max_caption_len,
                store_alphas=True,
                print_beam=print_beam,
            )
            print("Image COCO ID: {}".format(coco_id[0]))
            for caption, alpha in zip(top_k_generated_captions, alphas):
                visualize_attention(
                    image_features.squeeze(0), caption, alpha, word_map, smoothen=True
                )
        else:
            top_k_generated_captions = decoder.beam_search(
                encoded_features, beam_size, max_caption_len, print_beam=print_beam
            )

        generated_captions.append(top_k_generated_captions)

        coco_ids.append(coco_id[0])

        assert len(target_captions) == len(generated_captions)

    # Calculate metric scores
    for metric in metrics:
        metric_score = calculate_metric(
            metric,
            target_captions,
            generated_captions,
            coco_ids,
            word_map,
            occurrences_data,
        )
        print("\n{} score @ beam size {} is {}".format(metric, beam_size, metric_score))


def calculate_metric(
    metric_name,
    target_captions,
    generated_captions,
    coco_ids,
    word_map,
    occurrences_data,
):
    if metric_name == "bleu4":
        generated_captions = [
            get_caption_without_special_tokens(top_k_captions[0], word_map)
            for top_k_captions in generated_captions
        ]
        return corpus_bleu(target_captions, generated_captions)
    elif metric_name == "recall":
        return recall_adjective_noun_pairs(
            generated_captions, coco_ids, word_map, occurrences_data
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
        help="File containing occurrences statistics about adjective noun pairs",
        default="data/brown_dog.json",
    )
    parser.add_argument(
        "--checkpoint", help="Path to checkpoint of trained model", required=True
    )
    parser.add_argument(
        "--metrics",
        help="Evaluation metrics",
        nargs="+",
        default=["bleu4"],
        choices=["bleu4", "recall"],
    )

    parser.add_argument(
        "--beam-size", help="Size of the decoding beam", type=int, default=1
    )
    parser.add_argument(
        "--max-caption-len", help="Maximum caption length", type=int, default=50
    )
    parser.add_argument(
        "--visualize-attention",
        help="Visualize the attention for every sample",
        default=False,
        action="store_true",
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
    evaluate(
        data_folder=parsed_args.data_folder,
        occurrences_data=parsed_args.occurrences_data,
        checkpoint=parsed_args.checkpoint,
        metrics=parsed_args.metrics,
        beam_size=parsed_args.beam_size,
        max_caption_len=parsed_args.max_caption_len,
        visualize=parsed_args.visualize_attention,
        print_beam=parsed_args.print_beam,
    )
