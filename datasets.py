import torch
from torch.utils.data import Dataset
import h5py
import json
import os
import numpy as np

from utils import IMAGES_FILENAME, CAPTIONS_FILENAME, CAPTION_LENGTHS_FILENAME


class CaptionDataset(Dataset):
    """
    PyTorch Dataset that provides batches of images of a given split
    """

    def __init__(self, data_folder, split, transform=None):
        """
        :param data_folder: folder where data files are stored
        :param split: split, indices of images that should be included
        :param transform: pytorch image transform pipeline
        """
        self.h5py_file = h5py.File(os.path.join(data_folder, IMAGES_FILENAME), "r")

        self.captions_per_image = self.h5py_file.attrs["captions_per_image"]
        self.max_caption_len = self.h5py_file.attrs["max_caption_len"]

        self.split = split

        # Load references to images
        self.imgs = self.h5py_file["images"]

        # Load captions
        with open(os.path.join(data_folder, CAPTIONS_FILENAME), "r") as json_file:
            self.captions = json.load(json_file)

        # Load caption lengths
        with open(
            os.path.join(data_folder, CAPTION_LENGTHS_FILENAME), "r"
        ) as json_file:
            self.caption_lengths = json.load(json_file)

        # Set pytorch transformation pipeline
        self.transform = transform

        # Set size of the dataset
        self.dataset_size = len(self.split)

    def get_image_data(self, index):
        image_data = self.imgs[index]

        # normalize the values to be between [0,1]
        image_data = image_data / 255.0

        image = torch.FloatTensor(image_data)
        if self.transform:
            image = self.transform(image)

        return image

    def __getitem__(self, i):
        raise NotImplementedError

    def __len__(self):
        return self.dataset_size


class CaptionTrainDataset(CaptionDataset):
    """
    PyTorch training dataset that provides batches of images with a corresponding caption each.
    """

    def __init__(self, data_folder, split, transform=None):
        """
        :param data_folder: folder where data files are stored
        :param split: split, indices of images that should be included
        :param transform: pytorch image transform pipeline
        """

        h5py_file = h5py.File(os.path.join(data_folder, IMAGES_FILENAME), "r")

        captions_per_image = h5py_file.attrs["captions_per_image"]
        # convert list of indices of image to list of indices of captions (include all captions for each image)
        split = np.array(
            [
                np.arange(
                    i * captions_per_image, i * captions_per_image + captions_per_image
                )
                for i in split
            ]
        ).flatten()

        super(CaptionTrainDataset, self).__init__(data_folder, split, transform)

    def __getitem__(self, i):
        # Convert index depending on the dataset split
        converted_index = self.split[i]

        image = self.get_image_data(converted_index // self.captions_per_image)

        caption = torch.LongTensor(self.captions[converted_index])
        caption_length = torch.LongTensor([self.caption_lengths[converted_index]])

        return image, caption, caption_length


class CaptionTestDataset(CaptionDataset):
    """
    PyTorch test dataset that provides batches of images and all their corresponding captions.

    """

    def __init__(self, data_folder, split, transform=None):
        super(CaptionTestDataset, self).__init__(data_folder, split, transform)

    def __getitem__(self, i):
        # Convert index depending on the dataset split
        converted_index = self.split[i]

        image = self.get_image_data(converted_index)

        # Collect all available captions for the image and return them
        first_caption_index = converted_index * self.captions_per_image
        last_caption_index = first_caption_index + self.captions_per_image
        all_captions_for_image = torch.LongTensor(
            self.captions[first_caption_index:last_caption_index]
        )
        return image, all_captions_for_image
