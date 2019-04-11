import argparse
import json
import os
import sys
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision.transforms import transforms

from metrics import recall_captions_from_images
from models.bottom_up_top_down import TopDownDecoder
from models.captioning_model import create_encoder_optimizer, create_decoder_optimizer
from models.ranking_generating import RankGenDecoder, RankGenEncoder
from models.show_attend_tell import Encoder, SATDecoder
from datasets import CaptionTrainDataset, CaptionTestDataset
from nltk.translate.bleu_score import corpus_bleu

from utils import (
    adjust_learning_rate,
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
    get_splits,
)

MODEL_SHOW_ATTEND_TELL = "SHOW_ATTEND_TELL"
MODEL_BOTTOM_UP_TOP_DOWN = "BOTTOM_UP_TOP_DOWN"
MODEL_RANKING_GENERATING = "RANKING_GENERATING"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True  # improve performance if inputs to model are fixed size

epochs_since_last_improvement = 0
best_validation_metric_score = 0.0


def main(
    model_params,
    model_name,
    data_folder,
    occurrences_data,
    karpathy_json,
    batch_size,
    alpha_c,
    embeddings_file,
    grad_clip,
    fine_tune_encoder=False,
    workers=1,
    start_epoch=0,
    epochs=120,
    epochs_early_stopping=10,
    epochs_adjust_learning_rate=8,
    rate_adjust_learning_rate=0.8,
    val_set_size=0.1,
    checkpoint=None,
    print_freq=100,
):

    global best_validation_metric_score, epochs_since_last_improvement

    print("Starting training on device: ", device)

    # Read word map
    word_map_file = os.path.join(data_folder, WORD_MAP_FILENAME)
    with open(word_map_file, "r") as json_file:
        word_map = json.load(json_file)

    # Read pretrained word embeddings
    embeddings = None
    if embeddings_file:
        embeddings, model_params["embeddings_size"] = load_embeddings(
            embeddings_file, word_map
        )
        print(
            "Set embedding layer dimension to {}".format(
                model_params["embeddings_size"]
            )
        )

    # Generate dataset splits
    train_images_split, val_images_split, _ = get_splits(
        occurrences_data, karpathy_json, val_set_size
    )

    # Load checkpoint
    if checkpoint:
        checkpoint = torch.load(checkpoint, map_location=device)
        start_epoch = checkpoint["epoch"] + 1
        epochs_since_last_improvement = checkpoint["epochs_since_improvement"]
        best_validation_metric_score = checkpoint["validation_metric_score"]
        decoder = checkpoint["decoder"]
        decoder_optimizer = checkpoint["decoder_optimizer"]
        model_name = checkpoint["model_name"]
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

        elif model_name == MODEL_RANKING_GENERATING:
            encoder = RankGenEncoder(model_params)
            encoder_optimizer = create_encoder_optimizer(encoder, model_params)
            decoder = RankGenDecoder(word_map, model_params, embeddings)
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

    elif (
        model_name == MODEL_BOTTOM_UP_TOP_DOWN or model_name == MODEL_RANKING_GENERATING
    ):
        train_images_loader = torch.utils.data.DataLoader(
            CaptionTrainDataset(
                data_folder, BOTTOM_UP_FEATURES_FILENAME, train_images_split
            ),
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=True,
        )
        val_images_loader = torch.utils.data.DataLoader(
            CaptionTestDataset(
                data_folder, BOTTOM_UP_FEATURES_FILENAME, val_images_split
            ),
            batch_size=1,
            shuffle=True,
            num_workers=workers,
            pin_memory=True,
        )

    # Print configuration
    if encoder:
        print("Encoder params: {}".format(encoder.params))

    print("Decoder params: {}".format(decoder.params))

    # Move to GPU, if available
    decoder = decoder.to(device)
    if encoder:
        encoder.to(device)

    # Loss function
    loss_function = nn.CrossEntropyLoss().to(device)

    for epoch in range(start_epoch, epochs):
        if epochs_since_last_improvement >= epochs_early_stopping:
            print(
                "No improvement since {} epochs, stopping training".format(
                    epochs_since_last_improvement
                )
            )
            break
        if (
            epochs_since_last_improvement > 0
            and epochs_since_last_improvement % epochs_adjust_learning_rate == 0
        ):
            adjust_learning_rate(decoder_optimizer, rate_adjust_learning_rate)
            if fine_tune_encoder:
                adjust_learning_rate(encoder_optimizer, rate_adjust_learning_rate)

        # One epoch's training
        train(
            model_name,
            train_images_loader,
            encoder,
            decoder,
            encoder_optimizer,
            decoder_optimizer,
            loss_function,
            epoch,
            grad_clip,
            alpha_c,
            print_freq,
        )

        # One epoch's validation
        if model_name == MODEL_RANKING_GENERATING:
            current_validation_metric_score = validate_ranking(
                val_images_loader, encoder, decoder, val_images_split, print_freq
            )
        else:
            current_validation_metric_score = validate(
                val_images_loader, encoder, decoder, word_map, print_freq
            )
        # Check if there was an improvement
        current_checkpoint_is_best = (
            current_validation_metric_score > best_validation_metric_score
        )
        if current_checkpoint_is_best:
            best_validation_metric_score = current_validation_metric_score
            epochs_since_last_improvement = 0
        else:
            epochs_since_last_improvement += 1
            print(
                "\nEpochs since last improvement: {}".format(
                    epochs_since_last_improvement
                )
            )
            print("Best validation score: {}\n".format(best_validation_metric_score))

        # Save checkpoint
        if occurrences_data:
            name = os.path.basename(occurrences_data).split(".")[0]
        elif karpathy_json:
            name = "karpathy"
        save_checkpoint(
            model_name,
            name,
            epoch,
            epochs_since_last_improvement,
            encoder,
            decoder,
            encoder_optimizer,
            decoder_optimizer,
            current_validation_metric_score,
            current_checkpoint_is_best,
        )

    print("\n\nFinished training.")


def calculate_loss(
    model_name, loss_function, packed_scores, packed_targets, alphas, alpha_c
):
    if model_name == MODEL_BOTTOM_UP_TOP_DOWN or model_name == MODEL_RANKING_GENERATING:
        return loss_function(packed_scores, packed_targets)

    elif model_name == MODEL_SHOW_ATTEND_TELL:
        loss = loss_function(packed_scores, packed_targets)

        # Add doubly stochastic attention regularization
        loss += alpha_c * ((1.0 - alphas.sum(dim=1)) ** 2).mean()
        return loss


def train(
    model_name,
    data_loader,
    encoder,
    decoder,
    encoder_optimizer,
    decoder_optimizer,
    loss_function,
    epoch,
    grad_clip,
    alpha_c,
    print_freq,
):
    """
    Perform one training epoch.

    """

    decoder.train()
    if encoder:
        encoder.train()

    losses = AverageMeter()  # losses (per decoded word)

    # Loop over training batches
    for i, (images, target_captions, caption_lengths) in enumerate(data_loader):
        target_captions = target_captions.to(device)
        caption_lengths = caption_lengths.to(device)
        images = images.to(device)

        # Forward propagation
        if encoder:
            images = encoder(images)
        decode_lengths = caption_lengths.squeeze(1) - 1

        if model_name == MODEL_RANKING_GENERATING:
            images_embedded, captions_embedded = decoder.forward_ranking(
                images, target_captions, decode_lengths
            )
            loss = decoder.criterion(images_embedded, captions_embedded)

        else:
            scores, decode_lengths, alphas = decoder(
                images, target_captions, decode_lengths
            )

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            target_captions = target_captions[:, 1:]

            # Remove timesteps that we didn't decode at, or are pads
            decode_lengths, sort_ind = decode_lengths.sort(dim=0, descending=True)
            packed_scores, _ = pack_padded_sequence(
                scores[sort_ind], decode_lengths, batch_first=True
            )
            packed_targets, _ = pack_padded_sequence(
                target_captions[sort_ind], decode_lengths, batch_first=True
            )

            # Calculate loss
            loss = calculate_loss(
                model_name,
                loss_function,
                packed_scores,
                packed_targets,
                alphas,
                alpha_c,
            )

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

        # Print status
        if i % print_freq == 0:
            print(
                "Epoch: {0}[Batch {1}/{2}]\t"
                "Loss {loss.val:.4f} (Average: {loss.avg:.4f})\t".format(
                    epoch, i, len(data_loader), loss=losses
                )
            )

    print("\n * LOSS - {loss.avg:.3f}\n".format(loss=losses))


def validate(data_loader, encoder, decoder, word_map, print_freq):
    """
    Perform validation of one training epoch.

    """
    decoder.eval()
    if encoder:
        encoder.eval()

    target_captions = []
    generated_captions = []

    # Loop over batches
    for i, (images, all_captions_for_image, _) in enumerate(data_loader):
        images = images.to(device)

        # Forward propagation
        if encoder:
            images = encoder(images)
        scores, decode_lengths, alphas = decoder(images)

        if i % print_freq == 0:
            print("Validation: [Batch {0}/{1}]\t".format(i, len(data_loader)))

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

        assert len(target_captions) == len(generated_captions)

    bleu4 = corpus_bleu(target_captions, generated_captions)

    print("\n * BLEU-4 - {bleu}\n".format(bleu=bleu4))

    return bleu4


def validate_ranking(data_loader, encoder, decoder, testing_indices, print_freq):
    """
    Perform validation of one training epoch.

    """
    decoder.eval()
    encoder.eval()

    # Lists for target captions and generated captions for each image
    embedded_captions = {}
    embedded_images = {}

    for i, (image_features, captions, caption_lengths, coco_id) in enumerate(
        data_loader
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

        if i % print_freq == 0:
            print("Validation: [Batch {0}/{1}]\t".format(i, len(data_loader)))

    recalls_sum = recall_captions_from_images(
        embedded_images, embedded_captions, testing_indices
    )

    return recalls_sum


def check_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        help="Name of the model to be used",
        default=MODEL_SHOW_ATTEND_TELL,
        choices=[
            MODEL_SHOW_ATTEND_TELL,
            MODEL_BOTTOM_UP_TOP_DOWN,
            MODEL_RANKING_GENERATING,
        ],
    )
    parser.add_argument(
        "--data-folder",
        help="Folder where the preprocessed data is located",
        default=os.path.expanduser("../datasets/coco2014_preprocessed/"),
    )
    parser.add_argument(
        "--occurrences-data",
        help="File containing occurrences statistics about adjective noun pairs",
    )
    parser.add_argument(
        "--karpathy-json", help="File containing train/val/test split information"
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
        "--epochs", help="Maximum number of training epochs", type=int, default=120
    )
    parser.add_argument(
        "--embeddings",
        help="Path to a word GloVe embeddings file to be used to initialize the decoder word embeddings",
        default=None,
    )
    parser.add_argument("--grad-clip", help="Gradient clip", type=float, default=10.0)
    parser.add_argument(
        "--dont-fine-tune-embeddings",
        help="Do not fine tune the decoder embeddings",
        dest="fine_tune_decoder_embeddings",
        action="store_false",
    )

    parsed_args = parser.parse_args(args)
    print(parsed_args)
    return parsed_args


if __name__ == "__main__":
    parsed_args = check_args(sys.argv[1:])
    main(
        model_params=vars(parsed_args),
        model_name=parsed_args.model,
        data_folder=parsed_args.data_folder,
        occurrences_data=parsed_args.occurrences_data,
        karpathy_json=parsed_args.karpathy_json,
        batch_size=parsed_args.batch_size,
        alpha_c=parsed_args.alpha_c,
        embeddings_file=parsed_args.embeddings,
        grad_clip=parsed_args.grad_clip,
        checkpoint=parsed_args.checkpoint,
        epochs=parsed_args.epochs,
    )
