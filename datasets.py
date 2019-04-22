import torch
from torch.utils.data import Dataset
import h5py
import json
import os

from utils import IMAGES_META_FILENAME, DATA_CAPTIONS, DATA_CAPTION_LENGTHS


class CaptionDataset(Dataset):
    """
    PyTorch Dataset that provides batches of images of a given split
    """

    def __init__(
        self,
        data_folder,
        features_filename,
        split,
        normalize=None,
        features_scale_factor=1,
    ):
        """
        :param data_folder: folder where data files are stored
        :param features_filename: Filename of the image features file
        :param split: split, indices of images that should be included
        :param normalize: PyTorch normalization transformation
        :param features_scale_factor: Additional scale factor, applied before normalization
        """
        self.image_features = h5py.File(
            os.path.join(data_folder, features_filename), "r"
        )

        self.split = split
        self.features_scale_factor = features_scale_factor

        # Load image meta data, including captions
        with open(os.path.join(data_folder, IMAGES_META_FILENAME), "r") as json_file:
            self.images_meta = json.load(json_file)

        self.captions_per_image = len(
            next(iter(self.images_meta.values()))[DATA_CAPTIONS]
        )

        # Set pytorch transformation pipeline
        self.transform = normalize

        # Set size of the dataset
        self.dataset_size = len(self.split)

    def get_image_features(self, coco_id):
        image_data = self.image_features[coco_id][()]

        # scale the features with given factor
        image_data = image_data * self.features_scale_factor

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

        image = self.get_image_features(coco_id)
        caption = torch.LongTensor(
            self.images_meta[coco_id][DATA_CAPTIONS][caption_index]
        )
        caption_length = torch.LongTensor(
            [self.images_meta[coco_id][DATA_CAPTION_LENGTHS][caption_index]]
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

        image = self.get_image_features(coco_id)
        all_captions_for_image = torch.LongTensor(
            self.images_meta[coco_id][DATA_CAPTIONS]
        )
        caption_lengths = torch.LongTensor(
            self.images_meta[coco_id][DATA_CAPTION_LENGTHS]
        )

        return image, all_captions_for_image, caption_lengths, coco_id
