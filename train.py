import logging

import argparse
import json
import os
import sys
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torchvision.transforms import transforms

from eval import evaluate, METRIC_RECALL, METRIC_BLEU
from models.bottom_up_top_down import TopDownDecoder
from models.captioning_model import create_encoder_optimizer, create_decoder_optimizer
from models.show_attend_tell import Encoder, SATDecoder
from datasets import CaptionTrainDataset, CaptionTestDataset
from nltk.translate.bleu_score import corpus_bleu

from utils import (
    save_checkpoint,
    AverageMeter,
    clip_gradients,
    WORD_MAP_FILENAME,
    get_caption_without_special_tokens,
    IMAGENET_IMAGES_MEAN,
    IMAGENET_IMAGES_STD,
    BOTTOM_UP_FEATURES_FILENAME,
    IMAGES_FILENAME,
    load_embeddings,
    get_checkpoint_file_path,
    MODEL_SHOW_ATTEND_TELL,
    MODEL_BOTTOM_UP_TOP_DOWN,
    get_train_log_file_path,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True  # improve performance if inputs to model are fixed size


def main(
    model_params,
    model_name,
    data_folder,
    dataset_splits,
    batch_size,
    embeddings_file,
    grad_clip,
    epochs,
    name_suffix,
    fine_tune_encoder,
    workers=1,
    start_epoch=0,
    epochs_early_stopping=5,
    checkpoint=None,
    print_freq=100,
):
    epochs_since_last_improvement = 0
    best_generation_metric_score = 0.0

    # Get the dataset splits
    dataset_splits_dict = json.load(open(dataset_splits, "r"))
    train_images_split = dataset_splits_dict["train_images_split"]
    val_images_split = dataset_splits_dict["val_images_split"]

    # Load checkpoint
    if checkpoint:
        checkpoint = torch.load(checkpoint, map_location=device)
        start_epoch = checkpoint["epoch"] + 1
        epochs_since_last_improvement = checkpoint["epochs_since_improvement"]
        best_generation_metric_score = checkpoint["generation_metric_score"]
        decoder = checkpoint["decoder"]
        decoder_optimizer = checkpoint["decoder_optimizer"]
        model_name = checkpoint["model_name"]
        word_map = decoder.word_map
        if "encoder" in checkpoint and checkpoint["encoder"]:
            encoder = checkpoint["encoder"]
            encoder_optimizer = checkpoint["encoder_optimizer"]
        else:
            encoder = None
            encoder_optimizer = None
        if fine_tune_encoder and encoder_optimizer is None:
            encoder.set_fine_tuning_enabled(fine_tune_encoder)
            encoder_optimizer = create_encoder_optimizer(encoder, model_params)

    # No checkpoint given, initialize the model
    else:
        # Read word map
        word_map_file = os.path.join(data_folder, WORD_MAP_FILENAME)
        word_map = json.load(open(word_map_file, "r"))

        # Read pretrained word embeddings
        embeddings = None
        if embeddings_file:
            embeddings, model_params["word_embeddings_size"] = load_embeddings(
                embeddings_file, word_map
            )
            logging.info(
                "Setting embedding layer dimension to %d",
                model_params["word_embeddings_size"],
            )

        if model_name == MODEL_SHOW_ATTEND_TELL:
            decoder = SATDecoder(word_map, model_params, embeddings)
            decoder_optimizer = create_decoder_optimizer(decoder, model_params)
            encoder = Encoder(model_params)
            encoder_optimizer = (
                create_encoder_optimizer(encoder, model_params)
                if fine_tune_encoder
                else None
            )

        elif model_name == MODEL_BOTTOM_UP_TOP_DOWN:
            encoder = None
            encoder_optimizer = None
            decoder = TopDownDecoder(word_map, model_params, embeddings)
            decoder_optimizer = create_decoder_optimizer(decoder, model_params)

        else:
            raise RuntimeError("Unknown model name: {}".format(model_name))

    # Data loaders
    if model_name == MODEL_SHOW_ATTEND_TELL:
        normalize = transforms.Normalize(
            mean=IMAGENET_IMAGES_MEAN, std=IMAGENET_IMAGES_STD
        )
        train_images_loader = torch.utils.data.DataLoader(
            CaptionTrainDataset(
                data_folder,
                IMAGES_FILENAME,
                train_images_split,
                transforms.Compose([normalize]),
                features_scale_factor=1 / 255.0,
            ),
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=True,
        )
        val_images_loader = torch.utils.data.DataLoader(
            CaptionTestDataset(
                data_folder,
                IMAGES_FILENAME,
                val_images_split,
                transforms.Compose([normalize]),
                features_scale_factor=1 / 255.0,
            ),
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=True,
        )

    elif model_name == MODEL_BOTTOM_UP_TOP_DOWN:
        train_images_loader = torch.utils.data.DataLoader(
            CaptionTrainDataset(
                data_folder, BOTTOM_UP_FEATURES_FILENAME, train_images_split
            ),
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=True,
        )
        validation_batch_size = batch_size
        val_images_loader = torch.utils.data.DataLoader(
            CaptionTestDataset(
                data_folder, BOTTOM_UP_FEATURES_FILENAME, val_images_split
            ),
            batch_size=validation_batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=True,
        )

    # Log configuration
    if encoder:
        logging.info("Encoder params: %s", encoder.params)
    logging.info("Decoder params: %s", decoder.params)

    logging.info("Starting training on device: %s", device)

    # Move to GPU, if available
    if encoder:
        encoder.to(device)
    decoder = decoder.to(device)

    for epoch in range(start_epoch, epochs):
        if epochs_since_last_improvement >= epochs_early_stopping:
            logging.info(
                "No improvement since %d epochs, stopping training".format(
                    epochs_since_last_improvement
                )
            )
            break

        # One epoch's training
        train(
            train_images_loader,
            encoder,
            decoder,
            encoder_optimizer,
            decoder_optimizer,
            epoch,
            grad_clip,
            print_freq,
        )

        # One epoch's validation
        current_generation_metric_score = validate(
            val_images_loader, encoder, decoder, word_map, print_freq
        )
        current_checkpoint_is_best = (
            current_generation_metric_score > best_generation_metric_score
        )

        if current_checkpoint_is_best:
            best_generation_metric_score = current_generation_metric_score
            epochs_since_last_improvement = 0
        else:
            epochs_since_last_improvement += 1
            logging.info(
                "\nEpochs since last improvement: %d", epochs_since_last_improvement
            )
            logging.info("Best generation score: %d\n", best_generation_metric_score)

        # Save checkpoint
        save_checkpoint(
            model_name,
            dataset_splits,
            epoch,
            epochs_since_last_improvement,
            encoder,
            decoder,
            encoder_optimizer,
            decoder_optimizer,
            current_generation_metric_score,
            current_checkpoint_is_best,
            name_suffix,
        )

    logging.info("\n\nFinished training.")

    logging.info("Evaluating:")
    checkpoint_path = get_checkpoint_file_path(
        model_name, dataset_splits, name_suffix, True
    )
    metrics = [METRIC_BLEU, METRIC_RECALL]
    beam_size = 5
    evaluate(
        data_folder, dataset_splits, checkpoint_path, metrics, beam_size, False, False
    )


def train(
    data_loader,
    encoder,
    decoder,
    encoder_optimizer,
    decoder_optimizer,
    epoch,
    grad_clip,
    print_freq,
):
    """
    Perform one training epoch.

    """

    decoder.train()
    if encoder:
        encoder.train()

    losses = AverageMeter()

    # Loop over training batches
    for i, (images, target_captions, caption_lengths) in enumerate(data_loader):
        target_captions = target_captions.to(device)
        caption_lengths = caption_lengths.to(device)
        images = images.to(device)

        # Forward propagation
        if encoder:
            images = encoder(images)
        decode_lengths = caption_lengths.squeeze(1) - 1

        scores, decode_lengths, alphas = decoder(
            images, target_captions, decode_lengths
        )
        loss = decoder.loss(scores, target_captions, decode_lengths, alphas)

        # Back propagation
        decoder_optimizer.zero_grad()
        if encoder_optimizer:
            encoder_optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if grad_clip:
            clip_gradients(decoder_optimizer, grad_clip)
            if encoder_optimizer:
                clip_gradients(encoder_optimizer, grad_clip)

        # Update weights
        decoder_optimizer.step()
        if encoder_optimizer:
            encoder_optimizer.step()

        # Keep track of metrics
        losses.update(loss.item(), sum(decode_lengths).item())

        # Log status
        if i % print_freq == 0:
            logging.info(
                "Epoch: {0}[Batch {1}/{2}]\t"
                "Loss {loss.val:.4f} (Average: {loss.avg:.4f})\t".format(
                    epoch, i, len(data_loader), loss=losses
                )
            )

    logging.info("\n * LOSS - {loss.avg:.3f}\n".format(loss=losses))


def validate(data_loader, encoder, decoder, word_map, print_freq):
    """
    Perform validation of one training epoch.

    """
    decoder.eval()
    if encoder:
        encoder.eval()

    target_captions = []
    generated_captions = []
    coco_ids = []

    # Loop over batches
    for i, (images, all_captions_for_image, _, coco_id) in enumerate(data_loader):
        images = images.to(device)

        # Forward propagation
        if encoder:
            images = encoder(images)
        scores, decode_lengths, alphas = decoder(images)

        if i % print_freq == 0:
            logging.info("Validation: [Batch {0}/{1}]\t".format(i, len(data_loader)))

        # Target captions
        for j in range(all_captions_for_image.shape[0]):
            img_captions = [
                get_caption_without_special_tokens(caption, word_map)
                for caption in all_captions_for_image[j].tolist()
            ]
            target_captions.append(img_captions)

        # Generated captions
        _, captions = torch.max(scores, dim=2)
        captions = [
            get_caption_without_special_tokens(caption, word_map)
            for caption in captions.tolist()
        ]
        generated_captions.extend(captions)

        coco_ids.append(coco_id[0])

        assert len(target_captions) == len(generated_captions)

    bleu4 = corpus_bleu(target_captions, generated_captions)

    logging.info("\n * BLEU-4 - {bleu}\n".format(bleu=bleu4))

    return bleu4


def check_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        help="Name of the model to be used",
        default=MODEL_SHOW_ATTEND_TELL,
        choices=[MODEL_SHOW_ATTEND_TELL, MODEL_BOTTOM_UP_TOP_DOWN],
    )
    parser.add_argument(
        "--data-folder",
        help="Folder where the preprocessed data is located",
        default=os.path.expanduser("../datasets/coco2014_preprocessed/"),
    )
    parser.add_argument(
        "--dataset-splits", help="Pickled file containing the dataset splits"
    )
    parser.add_argument("--batch-size", help="Batch size", type=int, default=32)
    parser.add_argument(
        "--teacher-forcing",
        help="Teacher forcing rate (used in the decoder)",
        type=float,
    )
    parser.add_argument(
        "--encoder-learning-rate",
        help="Initial learning rate for the encoder (used only if fine-tuning is enabled)",
        type=float,
    )
    parser.add_argument(
        "--decoder-learning-rate",
        help="Initial learning rate for the decoder",
        type=float,
    )
    parser.add_argument(
        "--fine-tune-encoder", help="Fine tune the encoder", action="store_true"
    )
    parser.add_argument(
        "--alpha-c",
        help="regularization parameter for doubly stochastic attention (used in Show, Attend Tell model loss)",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--dropout-ratio", help="Dropout ratio in the decoder", type=float
    )
    parser.add_argument(
        "--checkpoint",
        help="Path to checkpoint of previously trained model",
        default=None,
    )
    parser.add_argument(
        "--name-suffix",
        help="Extra suffix to add to the output file names on saving.",
        default="",
    )
    parser.add_argument(
        "--epochs", help="Maximum number of training epochs", type=int, default=120
    )
    parser.add_argument(
        "--embeddings",
        help="Path to a word GloVe embeddings file to be used to initialize the decoder word embeddings",
        default=None,
    )
    parser.add_argument("--grad-clip", help="Gradient clip", type=float, default=10.0)
    parser.add_argument(
        "--dont-fine-tune-word-embeddings",
        help="Do not fine tune the decoder word embeddings",
        dest="fine_tune_decoder_word_embeddings",
        action="store_false",
    )

    parsed_args = parser.parse_args(args)
    return parsed_args


if __name__ == "__main__":
    parsed_args = check_args(sys.argv[1:])
    logging.basicConfig(
        filename=get_train_log_file_path(
            parsed_args.model,
            parsed_args.dataset_splits,
            parsed_args.name_suffix,
            parsed_args.embeddings,
        ),
        level=logging.INFO,
    )
    logging.info(parsed_args)
    main(
        model_params=vars(parsed_args),
        model_name=parsed_args.model,
        data_folder=parsed_args.data_folder,
        dataset_splits=parsed_args.dataset_splits,
        batch_size=parsed_args.batch_size,
        embeddings_file=parsed_args.embeddings,
        grad_clip=parsed_args.grad_clip,
        checkpoint=parsed_args.checkpoint,
        epochs=parsed_args.epochs,
        name_suffix=parsed_args.name_suffix,
        fine_tune_encoder=parsed_args.fine_tune_encoder,
    )
