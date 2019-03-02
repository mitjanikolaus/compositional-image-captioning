import json
import os

import torch

import skimage.io as io
import matplotlib.pyplot as plt

TOKEN_UNKNOWN = '<unk>'
TOKEN_START = '<start>'
TOKEN_END = '<end>'
TOKEN_PADDING = '<pad>'

SPLIT_TRAIN = 'TRAIN'
SPLIT_VAL = 'VAL'
SPLIT_TEST = 'TEST'

# Normalization for images (cf. https://pytorch-zh.readthedocs.io/en/latest/torchvision/models.html)
IMAGENET_IMAGES_MEAN = [0.485, 0.456, 0.406]
IMAGENET_IMAGES_STD = [0.229, 0.224, 0.225]


def getWordMapFilename():
  return 'word_map.json'


def getImagesFilename():
  return 'images.hdf5'


def getImageCocoIdsFilename():
  return 'coco_ids.json'


def getCaptionsFilename():
  return 'captions.json'


def getCaptionLengthsFilename():
  return 'caption_lengths.json'


def getImageIndicesSplitsFromFile(data_folder, test_set_image_coco_ids_file, val_set_size=0):
  image_coco_ids_file = os.path.join(data_folder, getImageCocoIdsFilename())
  with open(image_coco_ids_file, 'r') as json_file:
    image_coco_ids = json.load(json_file)

  with open(test_set_image_coco_ids_file, 'r') as json_file:
    test_set_image_coco_ids = json.load(json_file)

  test_images_split = [image_coco_ids.index(coco_id) for coco_id in test_set_image_coco_ids]

  indices_without_test = list(set(range(len(image_coco_ids))) - set(test_images_split))

  train_val_split = int((1 - val_set_size) * len(indices_without_test))
  train_images_split = indices_without_test[:train_val_split]
  val_images_split = indices_without_test[train_val_split:]

  return train_images_split, val_images_split, test_images_split


def showImg(img):
  plt.imshow(img.transpose(1, 2, 0))
  plt.show()


def clip_gradient(optimizer, grad_clip):
  """
  Clips gradients computed during backpropagation to avoid explosion of gradients.

  :param optimizer: optimizer with the gradients to be clipped
  :param grad_clip: clip value
  """
  for group in optimizer.param_groups:
    for param in group['params']:
      if param.grad is not None:
        param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer, decoder_optimizer,
                    bleu4, is_best):
  """
  Saves model checkpoint.

  :param data_name: base name of processed dataset
  :param epoch: epoch number
  :param epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
  :param encoder: encoder model
  :param decoder: decoder model
  :param encoder_optimizer: optimizer to update encoder's weights, if fine-tuning
  :param decoder_optimizer: optimizer to update decoder's weights
  :param bleu4: validation BLEU-4 score for this epoch
  :param is_best: is this checkpoint the best so far?
  """
  state = {'epoch': epoch,
           'epochs_since_improvement': epochs_since_improvement,
           'bleu-4': bleu4,
           'encoder': encoder,
           'decoder': decoder,
           'encoder_optimizer': encoder_optimizer,
           'decoder_optimizer': decoder_optimizer}
  filename = 'checkpoint.pth.tar'
  torch.save(state, filename)
  # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
  if is_best:
    torch.save(state, 'best_' + filename)


class AverageMeter(object):
  """
  Keeps track of most recent, average, sum, and count of a metric.
  """

  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, shrink_factor):
  """
  Shrinks learning rate by a specified factor.

  :param optimizer: optimizer whose learning rate must be shrunk.
  :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
  """

  print("\nDECAYING learning rate.")
  for param_group in optimizer.param_groups:
    param_group['lr'] = param_group['lr'] * shrink_factor
  print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def accuracy(scores, targets, k):
  """
  Computes top-k accuracy, from predicted and true labels.

  :param scores: scores from the model
  :param targets: true labels
  :param k: k in top-k accuracy
  :return: top-k accuracy
  """

  batch_size = targets.size(0)
  _, ind = scores.topk(k, 1, True, True)
  correct = ind.eq(targets.view(-1, 1).expand_as(ind))
  correct_total = correct.view(-1).float().sum()  # 0D tensor
  return correct_total.item() * (100.0 / batch_size)
