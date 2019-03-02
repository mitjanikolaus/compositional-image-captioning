import argparse
import json
import os
import sys
import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from models import Encoder, DecoderWithAttention
from datasets import CaptionDataset
from nltk.translate.bleu_score import corpus_bleu

from utils import getWordMapFilename, SPLIT_TRAIN, SPLIT_VAL, adjust_learning_rate, save_checkpoint, \
  AverageMeter, clip_gradient, accuracy, getImageIndicesSplitsFromFile, IMAGENET_IMAGES_MEAN, IMAGENET_IMAGES_STD

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

epochs_since_last_improvement = 0  # number of epochs since the last improvement of BLEU validation score
best_bleu4 = 0.  # BLEU-4 score right now


def main(data_folder, test_set_image_coco_ids_file, emb_dim=512, attention_dim=512, decoder_dim=512, dropout=0.5,
         start_epoch=0, epochs=120,
         batch_size=32, workers=1, encoder_lr=1e-4, decoder_lr=4e-4, grad_clip=5., alpha_c=1.,
         fine_tune_encoder=False, epochs_early_stopping=20, epochs_adjust_learning_rate=8,
         rate_adjust_learning_rate=0.8, val_set_size=0.1, checkpoint=None, print_freq=100):
  """
  Training and validation.
  """

  global best_bleu4, epochs_since_last_improvement

  print("Starting training on device: ", device)

  # Read word map
  word_map_file = os.path.join(data_folder, getWordMapFilename())
  with open(word_map_file, 'r') as json_file:
    word_map = json.load(json_file)

  # Load checkpoint
  if checkpoint:
    checkpoint = torch.load(checkpoint)
    start_epoch = checkpoint['epoch'] + 1
    epochs_since_last_improvement = checkpoint['epochs_since_improvement']
    best_bleu4 = checkpoint['bleu-4']
    decoder = checkpoint['decoder']
    decoder_optimizer = checkpoint['decoder_optimizer']
    encoder = checkpoint['encoder']
    encoder_optimizer = checkpoint['encoder_optimizer']
    if fine_tune_encoder and encoder_optimizer is None:
      encoder.set_fine_tuning_enabled(fine_tune_encoder)
      encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                           lr=encoder_lr)

  # No checkpoint given, initialize the model
  else:
    decoder = DecoderWithAttention(attention_dim=attention_dim,
                                   embed_dim=emb_dim,
                                   decoder_dim=decoder_dim,
                                   vocab_size=len(word_map),
                                   dropout=dropout)
    decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                         lr=decoder_lr)
    encoder = Encoder()
    encoder.set_fine_tuning_enabled(fine_tune_encoder)
    encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                           lr=encoder_lr) if fine_tune_encoder else None

  # Move to GPU, if available
  decoder = decoder.to(device)
  encoder = encoder.to(device)

  # Loss function
  loss_function = nn.CrossEntropyLoss().to(device)

  # Generate dataset splits
  train_images_split, val_images_split, test_images_split = getImageIndicesSplitsFromFile(
    data_folder, test_set_image_coco_ids_file, val_set_size
  )

  normalize = transforms.Normalize(mean=IMAGENET_IMAGES_MEAN, std=IMAGENET_IMAGES_STD)
  train_images_loader = torch.utils.data.DataLoader(
    CaptionDataset(data_folder, train_images_split, SPLIT_TRAIN, transform=transforms.Compose([normalize])),
    batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
  val_images_loader = torch.utils.data.DataLoader(
    CaptionDataset(data_folder, val_images_split, SPLIT_VAL, transform=transforms.Compose([normalize])),
    batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

  for epoch in range(start_epoch, epochs):

    if epochs_since_last_improvement >= epochs_early_stopping:
      print("No improvement since {} epochs, stopping training".format(epochs_since_last_improvement))
      break
    if epochs_since_last_improvement > 0 and epochs_since_last_improvement % epochs_adjust_learning_rate == 0:
      adjust_learning_rate(decoder_optimizer, rate_adjust_learning_rate)
      if fine_tune_encoder:
        adjust_learning_rate(encoder_optimizer, rate_adjust_learning_rate)

    # One epoch's training
    train(train_images_loader,
          encoder,
          decoder,
          loss_function,
          encoder_optimizer,
          decoder_optimizer,
          epoch,
          grad_clip,
          alpha_c,
          print_freq)

    # One epoch's validation
    current_bleu4 = validate(val_images_loader,
                             encoder,
                             decoder,
                             loss_function,
                             word_map,
                             alpha_c,
                             print_freq)

    # Check if there was an improvement
    current_checkpoint_is_best = current_bleu4 > best_bleu4
    if current_checkpoint_is_best:
      best_bleu4 = current_bleu4
      epochs_since_last_improvement = 0
    else:
      epochs_since_last_improvement += 1
      print("\nEpochs since last improvement: {}\n".format(epochs_since_last_improvement))

    # Save checkpoint
    save_checkpoint(epoch, epochs_since_last_improvement, encoder, decoder, encoder_optimizer,
                    decoder_optimizer, current_bleu4, current_checkpoint_is_best)

  print("\n\nFinished training.")


def train(train_loader, encoder, decoder, loss_function, encoder_optimizer, decoder_optimizer, epoch, grad_clip,
          alpha_c, print_freq):
  """
  Performs one epoch's training.

  :param train_loader: DataLoader for training data
  :param encoder: encoder model
  :param decoder: decoder model
  :param loss_function: loss layer
  :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
  :param decoder_optimizer: optimizer to update decoder's weights
  :param epoch: epoch number
  """

  decoder.train()  # train mode (dropout and batchnorm is used)
  encoder.train()

  batch_time = AverageMeter()  # forward prop. + back prop. time
  data_time = AverageMeter()  # data loading time
  losses = AverageMeter()  # loss (per word decoded)
  top5accs = AverageMeter()  # top5 accuracy

  start = time.time()

  # Batches
  for i, (imgs, caps, caplens) in enumerate(train_loader):
    data_time.update(time.time() - start)

    # Move to GPU, if available
    imgs = imgs.to(device)
    caps = caps.to(device)
    caplens = caplens.to(device)

    # Forward prop.
    imgs = encoder(imgs)
    scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

    # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
    targets = caps_sorted[:, 1:]

    # Remove timesteps that we didn't decode at, or are pads
    # pack_padded_sequence is an easy trick to do this
    scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
    targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

    # Calculate loss
    loss = loss_function(scores, targets)

    # Add doubly stochastic attention regularization
    loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

    # Back prop.
    decoder_optimizer.zero_grad()
    if encoder_optimizer:
      encoder_optimizer.zero_grad()
    loss.backward()

    # Clip gradients
    if grad_clip:
      clip_gradient(decoder_optimizer, grad_clip)
      if encoder_optimizer:
        clip_gradient(encoder_optimizer, grad_clip)

    # Update weights
    decoder_optimizer.step()
    if encoder_optimizer:
      encoder_optimizer.step()

    # Keep track of metrics
    top5 = accuracy(scores, targets, 5)
    losses.update(loss.item(), sum(decode_lengths))
    top5accs.update(top5, sum(decode_lengths))
    batch_time.update(time.time() - start)

    start = time.time()

    # Print status
    if i % print_freq == 0:
      print('Epoch: [{0}][{1}/{2}]\t'
            'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                    batch_time=batch_time,
                                                                    data_time=data_time, loss=losses,
                                                                    top5=top5accs))


def validate(val_loader, encoder, decoder, criterion, word_map, alpha_c, print_freq):
  """
  Performs one epoch's validation.

  :param val_loader: DataLoader for validation data.
  :param encoder: encoder model
  :param decoder: decoder model
  :param criterion: loss layer
  :return: BLEU-4 score
  """
  decoder.eval()  # eval mode (no dropout or batchnorm)
  if encoder is not None:
    encoder.eval()

  batch_time = AverageMeter()
  losses = AverageMeter()
  top5accs = AverageMeter()

  start = time.time()

  references = list()  # references (true captions) for calculating BLEU-4 score
  hypotheses = list()  # hypotheses (predictions)

  # Batches
  for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader):

    # Move to device, if available
    imgs = imgs.to(device)
    caps = caps.to(device)
    caplens = caplens.to(device)

    # Forward prop.
    if encoder:
      imgs = encoder(imgs)
    scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

    # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
    targets = caps_sorted[:, 1:]

    # Remove timesteps that we didn't decode at, or are pads
    # pack_padded_sequence is an easy trick to do this
    scores_copy = scores.clone()
    scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
    targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

    # Calculate loss
    loss = criterion(scores, targets)

    # Add doubly stochastic attention regularization
    loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

    # Keep track of metrics
    losses.update(loss.item(), sum(decode_lengths))
    top5 = accuracy(scores, targets, 5)
    top5accs.update(top5, sum(decode_lengths))
    batch_time.update(time.time() - start)

    start = time.time()

    if i % print_freq == 0:
      print('Validation: [{0}/{1}]\t'
            'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time,
                                                                      loss=losses, top5=top5accs))

    # Store references (true captions), and hypothesis (prediction) for each image
    # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
    # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

    # References
    allcaps = allcaps[sort_ind]  # because images were sorted in the decoder
    for j in range(allcaps.shape[0]):
      img_caps = allcaps[j].tolist()
      img_captions = list(
        map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
            img_caps))  # remove <start> and pads
      references.append(img_captions)

    # Hypotheses
    _, preds = torch.max(scores_copy, dim=2)
    preds = preds.tolist()
    temp_preds = list()
    for j, p in enumerate(preds):
      temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
    preds = temp_preds
    hypotheses.extend(preds)

    assert len(references) == len(hypotheses)

  # Calculate BLEU-4 scores
  bleu4 = corpus_bleu(references, hypotheses)

  print(
    '\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n'.format(
      loss=losses,
      top5=top5accs,
      bleu=bleu4))

  return bleu4


def check_args(args):
  parser = argparse.ArgumentParser()
  parser.add_argument('-D', '--data-folder',
                      help='Folder where the preprocessed data is located',
                      default=os.path.expanduser('~/datasets/coco2014_preprocessed/'))
  parser.add_argument('-T', '--test-set_image-coco-ids-file',
                      help='File containing JSON-serialized list of image IDs for the test set',
                      default='data/white_cars.json')
  parser.add_argument('-E', '--encoder-learning-rate',
                      help='Initial learning rate for the encoder (used only if fine-tuning is enabled)',
                      type=float, default=1e-4)
  parser.add_argument('-L', '--decoder-learning-rate',
                      help='Initial learning rate for the decoder',
                      type=float, default=4e-4)
  parser.add_argument('-F', '--fine-tune-encoder',
                      help='Fine tune the encoder',
                      action='store_true')
  parser.add_argument('-A', '--alpha-c',
                      help='regularization parameter for doubly stochastic attention',
                      type=float, default=1.)
  parser.add_argument('-C', '--checkpoint',
                      help='Path to checkpoint of previously trained model',
                      default=None)

  parsed_args = parser.parse_args(args)
  print(parsed_args)
  return parsed_args


if __name__ == '__main__':
  parsed_args = check_args(sys.argv[1:])
  main(
    data_folder=parsed_args.data_folder,
    test_set_image_coco_ids_file=parsed_args.test_set_image_coco_ids_file,
    encoder_lr=parsed_args.encoder_learning_rate,
    decoder_lr=parsed_args.decoder_learning_rate,
    alpha_c=parsed_args.alpha_c,
    fine_tune_encoder=parsed_args.fine_tune_encoder,
    checkpoint=parsed_args.checkpoint,
  )

