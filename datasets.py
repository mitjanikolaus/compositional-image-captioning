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
    TOKEN_POS_START,
    TOKEN_PADDING,
)


def concat_pos_tags_and_caption(caption, pos_tags, max_caption_len, word_map):
    concatenated = [word_map[TOKEN_POS_START]] + pos_tags + caption

    concatenated += [word_map[TOKEN_PADDING]] * (max_caption_len - len(concatenated))
    return concatenated


class CaptionDataset(Dataset):
    """
    PyTorch Dataset that provides batches of images of a given split
    """

    def __init__(
        self,
        data_folder,
        features_filename,
        split,
        max_caption_len,
        word_map,
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
        self.max_caption_len = max_caption_len
        self.word_map = word_map

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
        concatenated_caption_length = (caption_length - 2) * 2 + 3
        concatenated_caption_length = torch.LongTensor([concatenated_caption_length])
        pos_tags = self.images_meta[coco_id][DATA_CAPTIONS_POS][caption_index]

        concatenated_caption = torch.LongTensor(
            concat_pos_tags_and_caption(
                caption, pos_tags, self.max_caption_len, self.word_map
            )
        )

        return image, concatenated_caption, concatenated_caption_length

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

        concatenated_captions = [
            concat_pos_tags_and_caption(
                caption, pos_tags, self.max_caption_len, self.word_map
            )
            for caption, pos_tags in zip(captions, pos_tags)
        ]
        concatenated_captions = torch.LongTensor(concatenated_captions)

        concatenated_caption_lengths = [
            (caption_length - 2) * 2 + 3
            for caption_length in self.images_meta[coco_id][DATA_CAPTION_LENGTHS]
        ]

        concatenated_caption_lengths = torch.LongTensor(concatenated_caption_lengths)

        return image, concatenated_captions, concatenated_caption_lengths, coco_id
