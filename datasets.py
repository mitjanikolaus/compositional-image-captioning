import torch
from torch.utils.data import Dataset
import h5py
import json
import os

from utils import CAPTIONS_FILENAME, CAPTION_LENGTHS_FILENAME


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

        # Load captions
        with open(os.path.join(data_folder, CAPTIONS_FILENAME), "r") as json_file:
            self.captions = json.load(json_file)

        self.captions_per_image = len(next(iter(self.captions.values())))

        # Load caption lengths
        with open(
            os.path.join(data_folder, CAPTION_LENGTHS_FILENAME), "r"
        ) as json_file:
            self.caption_lengths = json.load(json_file)

        # Set pytorch transformation pipeline
        self.transform = normalize

        # Set size of the dataset
        self.dataset_size = len(self.split)

    def get_image_features(self, coco_id):
        image_data = self.image_features[coco_id].value

        # scale the features with given factor
        image_data = image_data * self.features_scale_factor

        image = torch.FloatTensor(image_data)
        if self.transform:
            image = self.transform(image)

        return image

    def __getitem__(self, i):
        raise NotImplementedError

    def __len__(self):
        return 64  # self.dataset_size


class CaptionTrainDataset(CaptionDataset):
    """
    PyTorch training dataset that provides batches of images with a corresponding caption each.
    """

    def __getitem__(self, i):
        # Convert index depending on the dataset split
        coco_id = self.split[i // self.captions_per_image]
        caption_index = i % self.captions_per_image

        image = self.get_image_features(coco_id)
        caption = torch.LongTensor(self.captions[coco_id][caption_index])
        caption_length = torch.LongTensor(
            [self.caption_lengths[coco_id][caption_index]]
        )

        return image, caption, caption_length

    def __len__(self):
        return 64  # self.dataset_size * self.captions_per_image


class CaptionTestDataset(CaptionDataset):
    """
    PyTorch test dataset that provides batches of images and all their corresponding captions.

    """

    def __getitem__(self, i):
        # Convert index depending on the dataset split
        coco_id = self.split[i]

        image = self.get_image_features(coco_id)
        all_captions_for_image = torch.LongTensor(self.captions[coco_id])

        return image, all_captions_for_image, coco_id
