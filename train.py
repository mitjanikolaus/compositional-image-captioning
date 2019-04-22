import argparse
import json
import os
import sys
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch import nn
from torchvision.transforms import transforms

from metrics import recall_captions_from_images
from models.bottom_up_top_down import TopDownDecoder
from models.bottom_up_top_down_ranking import BottomUpTopDownRankingDecoder
from models.captioning_model import create_encoder_optimizer, create_decoder_optimizer
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
MODEL_BOTTOM_UP_TOP_DOWN_RANKING = "BOTTOM_UP_TOP_DOWN_RANKING"

OBJECTIVE_GENERATION = "GENERATION"
OBJECTIVE_RANKING = "RANKING"
OBJECTIVE_JOINT = "JOINT"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True  # improve performance if inputs to model are fixed size

epochs_since_last_improvement = 0
best_ranking_metric_score = 0.0
best_generation_metric_score = 0.0


def main(
    model_params,
    model_name,
    data_folder,
    occurrences_data,
    karpathy_json,
    objective,
    batch_size,
    embeddings_file,
    grad_clip,
    epochs,
    checkpoint_suffix,
    fine_tune_decoder_image_embeddings,
    fine_tune_decoder_caption_embeddings,
    fine_tune_encoder,
    workers=1,
    start_epoch=0,
    epochs_early_stopping=10,
    epochs_adjust_learning_rate=8,
    rate_adjust_learning_rate=0.8,
    checkpoint=None,
    print_freq=100,
):

    global best_ranking_metric_score, best_generation_metric_score, epochs_since_last_improvement

    print("Starting training on device: ", device)

    # Generate dataset splits
    train_images_split, val_images_split, _ = get_splits(
        occurrences_data, karpathy_json
    )

    current_generation_metric_score = 0.0
    current_ranking_metric_score = 0.0

    # Load checkpoint
    if checkpoint:
        checkpoint = torch.load(checkpoint, map_location=device)
        start_epoch = checkpoint["epoch"] + 1
        epochs_since_last_improvement = checkpoint["epochs_since_improvement"]
        best_ranking_metric_score = checkpoint["ranking_metric_score"]
        best_generation_metric_score = checkpoint["generation_metric_score"]
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

        word_map = decoder.word_map

    # No checkpoint given, initialize the model
    else:
        # Read word map
        word_map_file = os.path.join(data_folder, WORD_MAP_FILENAME)
        with open(word_map_file, "r") as json_file:
            word_map = json.load(json_file)

        # Read pretrained word embeddings
        embeddings = None
        if embeddings_file:
            embeddings, model_params["word_embeddings_size"] = load_embeddings(
                embeddings_file, word_map
            )
            print(
                "Set embedding layer dimension to {}".format(
                    model_params["word_embeddings_size"]
                )
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

        elif model_name == MODEL_BOTTOM_UP_TOP_DOWN_RANKING:
            encoder = None
            encoder_optimizer = None
            decoder = BottomUpTopDownRankingDecoder(word_map, model_params, embeddings)
            decoder_optimizer = create_decoder_optimizer(decoder, model_params)

            if objective == OBJECTIVE_JOINT:
                loss_weight_generation = torch.ones(
                    1, requires_grad=True, device=device, dtype=torch.float
                )
                loss_weight_ranking = torch.ones(
                    1, requires_grad=True, device=device, dtype=torch.float
                )
                gradnorm_optimizer = torch.optim.Adam(
                    [loss_weight_generation, loss_weight_ranking], lr=0.025
                )  # TODO lr?
                gradnorm_loss = nn.L1Loss().to(device)

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
        model_name == MODEL_BOTTOM_UP_TOP_DOWN
        or model_name == MODEL_BOTTOM_UP_TOP_DOWN_RANKING
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
        validation_batch_size = batch_size
        if model_name == MODEL_BOTTOM_UP_TOP_DOWN_RANKING:
            validation_batch_size = 1
        val_images_loader = torch.utils.data.DataLoader(
            CaptionTestDataset(
                data_folder, BOTTOM_UP_FEATURES_FILENAME, val_images_split
            ),
            batch_size=validation_batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=True,
        )

    # Enable or disable training of image and caption embedding
    if model_name == MODEL_BOTTOM_UP_TOP_DOWN_RANKING:
        decoder.image_embedding.enable_fine_tuning(fine_tune_decoder_image_embeddings)
        decoder.language_encoding_lstm.enable_fine_tuning(
            fine_tune_decoder_caption_embeddings
        )

    # Print configuration
    if encoder:
        print("Encoder params: {}".format(encoder.params))
    print("Decoder params: {}".format(decoder.params))

    # Move to GPU, if available
    if encoder:
        encoder.to(device)
    decoder = decoder.to(device)

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
            epoch,
            objective,
            grad_clip,
            print_freq,
            gradnorm_optimizer,
            loss_weight_generation,
            loss_weight_ranking,
            gradnorm_loss,
        )

        # One epoch's validation
        if objective == OBJECTIVE_RANKING:
            current_ranking_metric_score = validate_ranking(
                val_images_loader, encoder, decoder, val_images_split, print_freq
            )
            current_checkpoint_is_best = (
                current_ranking_metric_score > best_ranking_metric_score
            )
        if objective == OBJECTIVE_GENERATION or objective == OBJECTIVE_JOINT:
            current_generation_metric_score = validate(
                val_images_loader, encoder, decoder, word_map, print_freq
            )
            current_checkpoint_is_best = (
                current_generation_metric_score > best_generation_metric_score
            )

        if current_checkpoint_is_best:
            best_generation_metric_score = current_generation_metric_score
            best_ranking_metric_score = current_ranking_metric_score
            epochs_since_last_improvement = 0
        else:
            epochs_since_last_improvement += 1
            print(
                "\nEpochs since last improvement: {}".format(
                    epochs_since_last_improvement
                )
            )
            print("Best ranking score: {}".format(best_ranking_metric_score))
            print("Best generation score: {}\n".format(best_generation_metric_score))

        # Save checkpoint
        save_checkpoint(
            model_name,
            occurrences_data,
            karpathy_json,
            epoch,
            epochs_since_last_improvement,
            encoder,
            decoder,
            encoder_optimizer,
            decoder_optimizer,
            current_ranking_metric_score,
            current_generation_metric_score,
            current_checkpoint_is_best,
            checkpoint_suffix,
        )

    print("\n\nFinished training.")


def train(
    model_name,
    data_loader,
    encoder,
    decoder,
    encoder_optimizer,
    decoder_optimizer,
    epoch,
    objective,
    grad_clip,
    print_freq,
    gradnorm_optimizer,
    loss_weight_generation,
    loss_weight_ranking,
    gradnorm_loss,
    gradnorm_alpha=0.16,  # TODO
):
    """
    Perform one training epoch.

    """

    loss_weights = [loss_weight_generation, loss_weight_ranking]

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

        if model_name == MODEL_BOTTOM_UP_TOP_DOWN_RANKING:
            if objective == OBJECTIVE_GENERATION:
                scores, decode_lengths, images_embedded, captions_embedded, alphas = decoder.forward_joint(
                    images, target_captions, decode_lengths
                )
                loss = decoder.loss(scores, target_captions, decode_lengths, alphas)
            elif objective == OBJECTIVE_JOINT:
                scores, decode_lengths, images_embedded, captions_embedded, alphas = decoder.forward_joint(
                    images, target_captions, decode_lengths
                )
                loss_generation = decoder.loss(
                    scores, target_captions, decode_lengths, alphas
                )
                loss_generation_weighted = loss_weights[0] * loss_generation
                loss_ranking = decoder.loss_ranking(images_embedded, captions_embedded)
                loss_ranking_weighted = loss_weights[1] * loss_ranking
                loss = (loss_generation_weighted + loss_ranking_weighted) / 2
            elif objective == OBJECTIVE_RANKING:
                images_embedded, captions_embedded = decoder.forward_ranking(
                    images, target_captions, decode_lengths
                )
                loss = decoder.loss_ranking(images_embedded, captions_embedded)

        else:
            scores, decode_lengths, alphas = decoder(
                images, target_captions, decode_lengths
            )
            loss = decoder.loss(scores, target_captions, decode_lengths, alphas)

        if objective == OBJECTIVE_JOINT:
            if epoch == 0:
                loss_generation_t0 = loss_generation
                loss_ranking_t0 = loss_ranking

            decoder_optimizer.zero_grad()
            if encoder_optimizer:
                encoder_optimizer.zero_grad()
            loss.backward(retain_graph=True)

            # Getting gradients of the first layers of each tower and calculate their l2-norm
            named_params = dict(decoder.named_parameters())
            shared_params = [
                param
                for param_name, param in named_params.items()
                if param_name in decoder.SHARED_PARAMS and param.requires_grad
            ]
            G1R = torch.autograd.grad(
                loss_generation, shared_params, retain_graph=True, create_graph=True
            )
            G1R_flattened = torch.cat([g.view(-1) for g in G1R])
            G1R_flattened = G1R_flattened.data
            G1 = torch.norm(loss_weights[0] * G1R_flattened, 2).unsqueeze(0)
            G2R = torch.autograd.grad(
                loss_ranking, shared_params, retain_graph=True, create_graph=True
            )
            G2R_flattened = torch.cat([g.view(-1) for g in G2R])
            G2R_flattened = G2R_flattened.data
            G2 = torch.norm(loss_weights[1] * G2R_flattened, 2).unsqueeze(0)
            G_avg = torch.div(torch.add(G1, G2), 2)

            # Calculating relative losses
            lhat1 = torch.div(loss_generation, loss_generation_t0)
            lhat2 = torch.div(loss_ranking, loss_ranking_t0)
            lhat_avg = torch.div(torch.add(lhat1, lhat2), 2)

            # Calculating relative inverse training rates for tasks
            inv_rate1 = torch.div(lhat1, lhat_avg)
            inv_rate2 = torch.div(lhat2, lhat_avg)

            # Calculating the constant target for Eq. 2 in the GradNorm paper
            C1 = G_avg * (inv_rate1) ** gradnorm_alpha
            C2 = G_avg * (inv_rate2) ** gradnorm_alpha
            C1 = C1.data
            C2 = C2.data

            gradnorm_optimizer.zero_grad()
            # Calculating the gradient loss according to Eq. 2 in the GradNorm paper
            Lgrad = torch.add(gradnorm_loss(G1, C1), gradnorm_loss(G2, C2))
            Lgrad.backward()

            # Updating loss weights
            gradnorm_optimizer.step()
        else:
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

        if objective == OBJECTIVE_JOINT:
            # Renormalizing the losses weights
            coef = 2 / torch.add(loss_weight_generation, loss_weight_ranking)
            loss_weights = [coef * loss_weight_generation, coef * loss_weight_ranking]
            print("Weights are:", loss_weight_generation, loss_weight_ranking)
            print("Renormalized weights are:", loss_weights)

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
    coco_ids = []

    # Loop over batches
    for i, (images, all_captions_for_image, _, coco_id) in enumerate(data_loader):
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

        coco_ids.append(coco_id[0])

        assert len(target_captions) == len(generated_captions)

    bleu4 = corpus_bleu(target_captions, generated_captions)

    print("\n * BLEU-4 - {bleu}\n".format(bleu=bleu4))

    return bleu4


def validate_ranking(data_loader, encoder, decoder, testing_indices, print_freq):
    """
    Perform validation of one training epoch.

    """
    decoder.eval()
    if encoder:
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

        if encoder:
            image_features = encoder(image_features)

        image_embedded, image_captions_embedded = decoder.forward_ranking(
            image_features, captions, decode_lengths
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
            MODEL_BOTTOM_UP_TOP_DOWN_RANKING,
        ],
    )
    parser.add_argument(
        "--data-folder",
        help="Folder where the preprocessed data is located",
        default=os.path.expanduser("../datasets/coco2014_preprocessed/"),
    )
    parser.add_argument(
        "--occurrences-data",
        nargs="+",
        help="Files containing occurrences statistics about adjective noun pairs",
    )
    parser.add_argument(
        "--karpathy-json", help="File containing train/val/test split information"
    )
    parser.add_argument(
        "--objective",
        help="Training objective for which the loss is calculated",
        default=OBJECTIVE_GENERATION,
        choices=[OBJECTIVE_GENERATION, OBJECTIVE_RANKING, OBJECTIVE_JOINT],
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
        "--checkpoint-suffix",
        help="Extra suffix to add to the checkpoint file name on saving.",
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
    parser.add_argument(
        "--dont-fine-tune-caption-embeddings",
        help="Do not fine tune the decoder caption embeddings",
        dest="fine_tune_decoder_caption_embeddings",
        action="store_false",
    )
    parser.add_argument(
        "--dont-fine-tune-image-embeddings",
        help="Do not fine tune the decoder image embeddings",
        dest="fine_tune_decoder_image_embeddings",
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
        objective=parsed_args.objective,
        batch_size=parsed_args.batch_size,
        embeddings_file=parsed_args.embeddings,
        grad_clip=parsed_args.grad_clip,
        checkpoint=parsed_args.checkpoint,
        epochs=parsed_args.epochs,
        checkpoint_suffix=parsed_args.checkpoint_suffix,
        fine_tune_decoder_image_embeddings=parsed_args.fine_tune_decoder_image_embeddings,
        fine_tune_decoder_caption_embeddings=parsed_args.fine_tune_decoder_caption_embeddings,
        fine_tune_encoder=parsed_args.fine_tune_encoder,
    )
