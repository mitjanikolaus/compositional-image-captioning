import argparse
import sys

import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from datasets import *
from inference import generate_caption
from metrics import adjective_noun_matches
from utils import *
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True  # improve performance if inputs to model are fixed size


def evaluate(
    data_folder,
    occurrences_data,
    checkpoint,
    metric=corpus_bleu,
    beam_size=1,
    max_caption_len=50,
):
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

    # Normalization
    normalize = transforms.Normalize(mean=IMAGENET_IMAGES_MEAN, std=IMAGENET_IMAGES_STD)

    # DataLoader
    _, _, test_images_split = get_splits_from_occurrences_data(occurrences_data)
    data_loader = torch.utils.data.DataLoader(
        CaptionTestDataset(
            data_folder, test_images_split, transform=transforms.Compose([normalize])
        ),
        batch_size=1,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
    )

    # Lists for target captions and generated captions for each image
    target_captions = []
    generated_captions = []

    for i, (image, all_captions_for_image) in enumerate(
        tqdm(data_loader, desc="Evaluate with beam size " + str(beam_size))
    ):

        # Target captions
        target_captions.append(
            [
                get_caption_without_special_tokens(caption, word_map)
                for caption in all_captions_for_image[0].tolist()
            ]
        )

        # Generated caption
        generated_caption = generate_caption(
            encoder,
            decoder,
            image,
            word_map,
            beam_size,
            max_caption_len,
            store_alphas=False,
        )
        generated_captions.append(
            get_caption_without_special_tokens(generated_caption, word_map)
        )

        # print(decode_caption(generated_caption, word_map))
        # show_img(image.squeeze(0).numpy())
        assert len(target_captions) == len(generated_captions)

    # Calculate metric scores
    metric_score = calculate_metric(
        metric, target_captions, generated_captions, word_map
    )

    print("\n{} score @ beam size {} is {}".format(metric, beam_size, metric_score))
    return metric_score


def calculate_metric(metric_name, target_captions, generated_captions, word_map):
    if metric_name == "bleu4":
        return corpus_bleu(target_captions, generated_captions)
    elif metric_name == "adj-n":
        return adjective_noun_matches(target_captions, generated_captions, word_map)


def check_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-D",
        "--data-folder",
        help="Folder where the preprocessed data is located",
        default=os.path.expanduser("~/datasets/coco2014_preprocessed/"),
    )
    parser.add_argument(
        "--occurrences-data",
        help="File containing occurrences statistics about adjective noun pairs",
        default="data/brown_dog.json",
    )
    parser.add_argument(
        "-C",
        "--checkpoint",
        help="Path to checkpoint of trained model",
        default="best_checkpoint.pth.tar",
    )
    parser.add_argument("--metric", help="Evaluation metric", default="bleu4")
    parser.add_argument(
        "-B", "--beam-size", help="Size of the decoding beam", type=int, default=1
    )
    parser.add_argument(
        "-L", "--max-caption-len", help="Maximum caption length", type=int, default=50
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
        metric=parsed_args.metric,
        beam_size=parsed_args.beam_size,
        max_caption_len=parsed_args.max_caption_len,
    )
