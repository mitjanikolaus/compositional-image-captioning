import json
import os
import numpy as np
from collections import Counter

import h5py
from imageio import imread
from tqdm import tqdm
from skimage.transform import resize

from utils import getWordMapFilename, getImagesFilename, getCaptionsFilename, getCaptionLengthsFilename, TOKEN_UNKNOWN, \
  TOKEN_START, TOKEN_END, TOKEN_PADDING, SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST


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
  img = resize(img, (256, 256), mode='constant', anti_aliasing=False)
  img = img.transpose(2, 0, 1)
  assert img.shape == (3, 256, 256)
  assert np.max(img) <= 255
  return img

def encodeCaption(caption, word_map, max_caption_len):
  return ([word_map[TOKEN_START]]
          + [word_map.get(word, word_map[TOKEN_UNKNOWN]) for word in caption]
          + [word_map[TOKEN_END]]
          + [word_map[TOKEN_PADDING]] * (max_caption_len - len(caption)))


def preprocessImagesAndCaptions(train_val_test_splits, images_folder, captions_per_image, vocabulary_size, output_folder,
                                max_caption_len):
  # Read JSON defining the splits
  with open(train_val_test_splits, 'r') as j:
    data = json.load(j)

  # Read image paths and captions for each image
  train_image_paths = []
  train_image_captions = []
  val_image_paths = []
  val_image_captions = []
  test_image_paths = []
  test_image_captions = []
  word_freq = Counter()

  for img in data['images']:
    captions = []
    for caption in img['sentences']:
      word_freq.update(caption['tokens'])
      if len(caption['tokens']) <= max_caption_len:
        captions.append(caption['tokens'])

    if len(captions) == 0:
      continue

    path = os.path.join(images_folder, img['filepath'], img['filename'])

    if img['split'] in {'train', 'restval'}:
      train_image_paths.append(path)
      train_image_captions.append(captions)
    elif img['split'] in {'val'}:
      val_image_paths.append(path)
      val_image_captions.append(captions)
    elif img['split'] in {'test'}:
      test_image_paths.append(path)
      test_image_captions.append(captions)

  # Select the most frequent x words, where x is vocabulary_size
  words = [w for w,c in word_freq.most_common(vocabulary_size)]

  # Create word mapping
  word_map = createWordMap(words)

  # Save the word mapping to JSON
  word_map_filename = getWordMapFilename()

  with open(os.path.join(output_folder,word_map_filename), 'w') as file:
    json.dump(word_map, file)

  # Save images to HDF5 file, captions and their lengths to JSON files
  for image_paths, image_captions, split in [(train_image_paths, train_image_captions, SPLIT_TRAIN),
                                 (val_image_paths, val_image_captions, SPLIT_VAL),
                                 (test_image_paths, test_image_captions, SPLIT_TEST)]:

    # Create hdf5 file and dataset
    images_filename = getImagesFilename(split)
    with h5py.File(os.path.join(output_folder, images_filename), 'a') as h5py_file:
      h5py_file.attrs['captions_per_image'] = captions_per_image

      image_dataset = h5py_file.create_dataset('images', (len(image_paths), 3, 256, 256), dtype='uint8')

      encoded_captions = []
      caption_lengths = []

      for i, path in enumerate(tqdm(image_paths)):

        # Discard any additional captions
        captions = image_captions[i][:captions_per_image]

        assert len(captions) == captions_per_image

        # Read image
        img = readImage(path)

        # Save image to HDF5 file
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
      captions_filename = getCaptionsFilename(split)
      with open(os.path.join(output_folder, captions_filename), 'w') as json_file:
        json.dump(encoded_captions, json_file)

      caption_lengths_filename = getCaptionLengthsFilename(split)
      with open(os.path.join(output_folder, caption_lengths_filename), 'w') as json_file:
        json.dump(caption_lengths, json_file)


if __name__ == '__main__':
  preprocessImagesAndCaptions(
    train_val_test_splits='/home/mitja/datasets/karpathy_json/dataset_coco.json',
    images_folder='/home/mitja/datasets/coco2014/',
    captions_per_image=5,
    vocabulary_size=10000,
    output_folder='/home/mitja/datasets/coco2014_preprocessed/',
    max_caption_len=50)
