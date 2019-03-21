import torch
from torch.utils.data import Dataset
import h5py
import json
import os

from utils import (
    IMAGES_META_FILENAME,
    DATA_CAPTIONS,
    DATA_CAPTION_LENGTHS,
    DATA_CAPTIONS_POS,
)


def interleave_caption_pos_tags(caption, pos_tags):
    interleaved = []
    for token, pos_tag in zip(caption, pos_tags):
        interleaved.append(token)
        interleaved.append(pos_tag)
    interleaved.append(caption[len(pos_tags) + 1])
    interleaved += [caption[-1]] * (len(caption) - len(interleaved) - 1)
    return interleaved


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
        caption = self.images_meta[coco_id][DATA_CAPTIONS][caption_index]
        caption_length = self.images_meta[coco_id][DATA_CAPTION_LENGTHS][caption_index]
        interleaved_caption_length = (caption_length - 2) * 2 + 2
        interleaved_caption_length = torch.LongTensor([interleaved_caption_length])
        pos_tags = self.images_meta[coco_id][DATA_CAPTIONS_POS][caption_index]

        interleaved_caption = torch.LongTensor(
            interleave_caption_pos_tags(caption, pos_tags)
        )

        return image, interleaved_caption, interleaved_caption_length

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
        captions = self.images_meta[coco_id][DATA_CAPTIONS]
        pos_tags = self.images_meta[coco_id][DATA_CAPTIONS_POS]

        interleaved_captions = [
            interleave_caption_pos_tags(caption, pos_tags)
            for caption, pos_tags in zip(captions, pos_tags)
        ]
        interleaved_captions = torch.LongTensor(interleaved_captions)

        return image, interleaved_captions, coco_id
