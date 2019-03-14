import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import h5py
import json
import os

from utils import (
    IMAGES_FILENAME,
    CAPTIONS_FILENAME,
    CAPTION_LENGTHS_FILENAME,
    BOTTOM_UP_FEATURES_FILENAME,
    IMAGENET_IMAGES_MEAN,
    IMAGENET_IMAGES_STD,
)


class CaptionDataset(Dataset):
    """
    PyTorch Dataset that provides batches of images of a given split
    """

    def __init__(self, data_folder, split):
        """
        :param data_folder: folder where data files are stored
        :param split: split, indices of images that should be included
        """
        self.h5py_file = h5py.File(os.path.join(data_folder, IMAGES_FILENAME), "r")

        self.captions_per_image = self.h5py_file.attrs["captions_per_image"]
        self.max_caption_len = self.h5py_file.attrs["max_caption_len"]

        self.split = split

        # Load captions
        with open(os.path.join(data_folder, CAPTIONS_FILENAME), "r") as json_file:
            self.captions = json.load(json_file)

        # Load caption lengths
        with open(
            os.path.join(data_folder, CAPTION_LENGTHS_FILENAME), "r"
        ) as json_file:
            self.caption_lengths = json.load(json_file)

        # Set pytorch transformation pipeline
        normalize = transforms.Normalize(
            mean=IMAGENET_IMAGES_MEAN, std=IMAGENET_IMAGES_STD
        )

        self.transform = transforms.Compose([normalize])

        # Set size of the dataset
        self.dataset_size = len(self.split)

    def get_image_data(self, coco_id):
        image_data = self.h5py_file[coco_id].value

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

    def __getitem__(self, i):
        # Convert index depending on the dataset split
        coco_id = self.split[i // self.captions_per_image]
        caption_index = i % self.captions_per_image

        image = self.get_image_data(coco_id)
        caption = torch.LongTensor(self.captions[coco_id][caption_index])
        caption_length = torch.LongTensor(
            [self.caption_lengths[coco_id][caption_index]]
        )

        return image, caption, caption_length

    def __len__(self):
        return self.dataset_size * self.captions_per_image


class CaptionTestDataset(CaptionDataset):
    """
    PyTorch test dataset that provides batches of images and all their corresponding captions.

    """

    def __getitem__(self, i):
        # Convert index depending on the dataset split
        coco_id = self.split[i]

        image = self.get_image_data(coco_id)
        all_captions_for_image = torch.LongTensor(self.captions[coco_id])

        return image, all_captions_for_image, coco_id


class BottomUpDataset(Dataset):
    """
    PyTorch Dataset that provides batches of image features of a given split
    """

    def __init__(self, data_folder, split, transform=None):
        """
        :param data_folder: folder where data files are stored
        :param split: split, indices of images that should be included
        :param transform: pytorch image transform pipeline
        """
        self.image_features_file = h5py.File(
            os.path.join(data_folder, BOTTOM_UP_FEATURES_FILENAME), "r"
        )

        self.captions_per_image = 5  # self.h5py_file.attrs["captions_per_image"]
        self.max_caption_len = 50  # self.h5py_file.attrs["max_caption_len"]

        self.split = split

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
        self.dataset_size = 64  ##len(self.split)

    def get_image_data(self, coco_id):
        image_data = self.image_features_file[coco_id].value

        image = torch.FloatTensor(image_data)
        if self.transform:
            image = self.transform(image)

        return image

    def __getitem__(self, i):
        raise NotImplementedError

    def __len__(self):
        return self.dataset_size


class BottumUpTrainDataset(BottomUpDataset):
    """
    PyTorch training dataset that provides batches of images with a corresponding caption each.
    """

    def __getitem__(self, i):
        # Convert index depending on the dataset split
        coco_id = self.split[i // self.captions_per_image]
        caption_index = i % self.captions_per_image

        image = self.get_image_data(coco_id)
        caption = torch.LongTensor(self.captions[coco_id][caption_index])
        caption_length = torch.LongTensor(
            [self.caption_lengths[coco_id][caption_index]]
        )

        return image, caption, caption_length

    def __len__(self):
        return 64  # self.dataset_size * self.captions_per_image


class BottomUpTestDataset(BottomUpDataset):
    """
    PyTorch test dataset that provides batches of images and all their corresponding captions.

    """

    def __getitem__(self, i):
        # Convert index depending on the dataset split
        coco_id = self.split[i]

        image = self.get_image_data(coco_id)
        all_captions_for_image = torch.LongTensor(self.captions[coco_id])

        return image, all_captions_for_image, coco_id
