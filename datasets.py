import torch
from torch.utils.data import Dataset
import h5py
import json
import os
import numpy as np

from utils import getImagesFilename, getCaptionsFilename, getCaptionLengthsFilename, SPLIT_TRAIN, showImg


class CaptionDataset(Dataset):
  """
  PyTorch Dataset that provides batches of images of a given split
  """

  def __init__(self, data_folder, split, split_type, transform=None):
    """
    :param data_folder: folder where data files are stored
    :param split: split, indices of images that should be included
    :param split_type, identifier for the split ('TRAIN', 'VAL', or 'TEST')
    :param transform: pytorch image transform pipeline
    """
    self.split_type = split_type
    self.h5py_file = h5py.File(os.path.join(data_folder, getImagesFilename()), 'r')

    self.captions_per_image = self.h5py_file.attrs['captions_per_image']
    self.max_caption_len = self.h5py_file.attrs['max_caption_len']

    # convert list of indices of image to list of indices of captions (include all caption for each image)
    self.split = np.array(
      [np.arange(i*self.captions_per_image,i*self.captions_per_image+self.captions_per_image) for i in split]
    ).flatten()

    # Load references to images
    self.imgs = self.h5py_file['images']

    # Load captions
    with open(os.path.join(data_folder, getCaptionsFilename()), 'r') as json_file:
      self.captions = json.load(json_file)

    # Load caption lengths
    with open(os.path.join(data_folder, getCaptionLengthsFilename()), 'r') as json_file:
      self.caption_lengths = json.load(json_file)

    # Set pytorch transformation pipeline
    self.transform = transform

    # Set size of the dataset
    self.dataset_size = len(self.split)

  def __getitem__(self, i):

    # Convert index depending on the dataset split
    converted_index = self.split[i]

    # Get the coresponding image for the caption
    image = self.imgs[converted_index // self.captions_per_image]

    # normalize the values to be between [0,1]
    image = image / 255.

    img = torch.FloatTensor(image)
    if self.transform:
      img = self.transform(img)

    caption = torch.LongTensor(self.captions[converted_index])
    caption_length = torch.LongTensor([self.caption_lengths[converted_index]])

    if self.split_type == SPLIT_TRAIN:
      return img, caption, caption_length
    else:
      # For validation and testing, return all captions for the image to calculate the metrics
      first_caption_index = (converted_index // self.captions_per_image) * self.captions_per_image
      last_caption_index = first_caption_index + self.captions_per_image
      all_captions = torch.LongTensor(self.captions[first_caption_index:last_caption_index])
      return img, caption, caption_length, all_captions

  def __len__(self):
    return self.dataset_size
