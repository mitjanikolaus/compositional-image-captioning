import argparse
import json
import os
import string
import sys

import numpy as np
from collections import Counter

import h5py
from nltk import word_tokenize
from pycocotools.coco import COCO
from scipy.misc import imread, imresize
from tqdm import tqdm

from utils import getWordMapFilename, getImagesFilename, getCaptionsFilename, getCaptionLengthsFilename, TOKEN_UNKNOWN, \
  TOKEN_START, TOKEN_END, TOKEN_PADDING, getImageCocoIdsFilename


def createWordMap(words):
  word_map = {w: i + 1 for i, w in enumerate(words)}
  # Mapping for special characters
  word_map[TOKEN_UNKNOWN] = len(word_map) + 1
  word_map[TOKEN_START] = len(word_map) + 1
  word_map[TOKEN_END] = len(word_map) + 1
  word_map[TOKEN_PADDING] = 0

  return word_map

def readImage(path):
  img = imread(path)
  if len(img.shape) == 2:  # b/w image
    img = img[:, :, np.newaxis]
    img = np.concatenate([img, img, img], axis=2)
  img = imresize(img, (256, 256))
  img = img.transpose(2, 0, 1)
  assert img.shape == (3, 256, 256)
  assert np.max(img) <= 255
  return img

def encodeCaption(caption, word_map, max_caption_len):
  return ([word_map[TOKEN_START]]
          + [word_map.get(word, word_map[TOKEN_UNKNOWN]) for word in caption]
          + [word_map[TOKEN_END]]
          + [word_map[TOKEN_PADDING]] * (max_caption_len - len(caption)))


def preprocessImagesAndCaptions(dataset_folder, output_folder, vocabulary_size, captions_per_image):
  data_type = 'train2014'

  annFile = '{}/annotations/captions_{}.json'.format(dataset_folder, data_type)
  coco = COCO(annFile)

  images = coco.loadImgs(coco.getImgIds())

  image_paths = []
  image_captions = []

  image_coco_ids = []

  word_freq = Counter()
  max_caption_len = 0

  for img in images:
    captions = []

    annIds = coco.getAnnIds(imgIds=[img['id']])
    anns = coco.loadAnns(annIds)
    for ann in anns:
      caption = ann['caption'].lower()

      # Remove special chars and punctuation
      caption = caption.replace('\n', '').replace('"', '')
      caption = caption.translate(str.maketrans(dict.fromkeys(string.punctuation)))

      # Tokenize the caption
      caption = word_tokenize(caption)

      word_freq.update(caption)
      captions.append(caption)

      if len(caption) > max_caption_len:
        max_caption_len = len(caption)

    path = os.path.join(dataset_folder, data_type, img['file_name'])

    image_paths.append(path)
    image_captions.append(captions)
    image_coco_ids.append(img['id'])

  # Save image coco ids to JSON file
  image_coco_ids_path = os.path.join(output_folder, getImageCocoIdsFilename())
  print("Saving image COCO IDs to {}".format(image_coco_ids_path))
  with open(image_coco_ids_path, 'w') as json_file:
    json.dump(image_coco_ids, json_file)

  # Select the most frequent words
  words = [w for w,c in word_freq.most_common(vocabulary_size)]

  # Create word map
  word_map = createWordMap(words)
  word_map_path = os.path.join(output_folder, getWordMapFilename())

  print("Saving word mapping to {}".format(word_map_path))
  with open(word_map_path, 'w') as file:
    json.dump(word_map, file)

  # Create hdf5 file and dataset for the images
  images_dataset_path = os.path.join(output_folder, getImagesFilename())
  print("Creating image dataset at {}".format(images_dataset_path))
  with h5py.File(images_dataset_path, 'a') as h5py_file:
    h5py_file.attrs['captions_per_image'] = captions_per_image
    h5py_file.attrs['max_caption_len'] = max_caption_len

    image_dataset = h5py_file.create_dataset('images', (len(image_paths), 3, 256, 256), dtype='uint8')

    encoded_captions = []
    caption_lengths = []

    for i, path in enumerate(tqdm(image_paths)):

      # Discard any additional captions
      captions = image_captions[i][:captions_per_image]

      assert len(captions) == captions_per_image

      # Read image and save it to hdf5 file
      img = readImage(path)
      image_dataset[i] = img

      for j, caption in enumerate(captions):
        # Encode caption
        encoded_caption = encodeCaption(caption, word_map, max_caption_len)
        encoded_captions.append(encoded_caption)

        # extend caption length by 2 for start and end of sentence tokens
        caption_length = len(caption) + 2
        caption_lengths.append(caption_length)

    # Sanity check
    assert image_dataset.shape[0] * captions_per_image == len(encoded_captions) == len(caption_lengths)

    # Save encoded captions and their lengths to JSON files
    captions_path = os.path.join(output_folder, getCaptionsFilename())
    print("Saving encoded captions to {}".format(captions_path))
    with open(captions_path, 'w') as json_file:
      json.dump(encoded_captions, json_file)
    caption_lengths_path = os.path.join(output_folder, getCaptionLengthsFilename())
    print("Saving caption lengths to {}".format(caption_lengths_path))
    with open(caption_lengths_path, 'w') as json_file:
      json.dump(caption_lengths, json_file)

def check_args(args):
  parser = argparse.ArgumentParser()
  parser.add_argument('-D', '--dataset-folder',
                      help='Folder where the coco dataset is located',
                      default=os.path.expanduser('~/datasets/coco2014/'))
  parser.add_argument('-O', '--output-folder',
                      help='Folder in which the preprocessed data should be stored',
                      default=os.path.expanduser('~/datasets/coco2014_preprocessed/'))
  parser.add_argument('-V', '--vocabulary-size',
                      help='Number of words that should be saved in the vocabulary',
                      default=10000)
  parser.add_argument('-C', '--captions-per-image',
                      help='Number of captions per image. Additional captions are discarded.',
                      default=5)

  parsed_args = parser.parse_args(args)
  print(parsed_args)
  return (parsed_args.dataset_folder,
          parsed_args.output_folder,
          parsed_args.vocabulary_size,
          parsed_args.captions_per_image)


if __name__ == '__main__':
  dataset_folder, output_folder, vocabulary_size, captions_per_image = check_args(sys.argv[1:])
  preprocessImagesAndCaptions(dataset_folder, output_folder, vocabulary_size, captions_per_image)
