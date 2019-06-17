import json
import logging
import os

import torch

from scipy.misc import imread, imresize
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import transforms
from tqdm import tqdm

TOKEN_UNKNOWN = "<unk>"
TOKEN_START = "<start>"
TOKEN_END = "<end>"
TOKEN_PADDING = "<pad>"

# Normalization for images (cf. https://pytorch-zh.readthedocs.io/en/latest/torchvision/models.html)
IMAGENET_IMAGES_MEAN = [0.485, 0.456, 0.406]
IMAGENET_IMAGES_STD = [0.229, 0.224, 0.225]


WORD_MAP_FILENAME = "word_map.json"
IMAGES_FILENAME = "images.hdf5"
TEST_IMAGES_FILENAME = "images_coco_test.hdf5"
BOTTOM_UP_FEATURES_FILENAME = "bottom_up_features.hdf5"
IMAGES_META_FILENAME = "images_meta.json"
POS_TAGGED_CAPTIONS_FILENAME = "pos_tagged_captions.p"

DATA_CAPTIONS = "captions"
DATA_CAPTION_LENGTHS = "caption_lengths"
DATA_COCO_SPLIT = "coco_split"


NOUNS = "nouns"
ADJECTIVES = "adjectives"
VERBS = "verbs"

OCCURRENCE_DATA = "adjective_noun_occurrence_data"
PAIR_OCCURENCES = "pair_occurrences"
NOUN_OCCURRENCES = "noun_occurrences"
ADJECTIVE_OCCURRENCES = "adjective_occurrences"
VERB_OCCURRENCES = "verb_occurrences"

RELATION_NOMINAL_SUBJECT = "nsubj"
RELATION_ADJECTIVAL_MODIFIER = "amod"
RELATION_CONJUNCT = "conj"
RELATION_RELATIVE_CLAUSE_MODIFIER = "acl:relcl"
RELATION_ADJECTIVAL_CLAUSE = "acl"
RELATION_OBJECT = "obj"
RELATION_INDIRECT_OBJECT = "iobj"


MODEL_SHOW_ATTEND_TELL = "SHOW_ATTEND_TELL"
MODEL_BOTTOM_UP_TOP_DOWN = "BOTTOM_UP_TOP_DOWN"
MODEL_BOTTOM_UP_TOP_DOWN_RANKING = "BOTTOM_UP_TOP_DOWN_RANKING"

base_dir = os.path.dirname(os.path.abspath(__file__))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_adjectives_for_noun(pos_tagged_caption, nouns):
    dependencies = pos_tagged_caption.dependencies

    adjectives = {
        d[2].lemma
        for d in dependencies
        if d[1] == RELATION_ADJECTIVAL_MODIFIER
        and d[0].lemma in nouns
        and d[2].upos == "ADJ"
    } | {
        d[0].lemma
        for d in dependencies
        if d[1] == RELATION_NOMINAL_SUBJECT
        and d[2].lemma in nouns
        and d[0].upos == "ADJ"
    }
    conjuncted_adjectives = set()
    for adjective in adjectives:
        conjuncted_adjectives.update(
            {
                d[2].lemma
                for d in dependencies
                if d[1] == RELATION_CONJUNCT
                and d[0].lemma == adjective
                and d[2].upos == "ADJ"
            }
            | {
                d[2].lemma
                for d in dependencies
                if d[1] == RELATION_ADJECTIVAL_MODIFIER
                and d[0].lemma == adjective
                and d[2].upos == "ADJ"
            }
        )
    return adjectives | conjuncted_adjectives


def get_verbs_for_noun(pos_tagged_caption, nouns):
    dependencies = pos_tagged_caption.dependencies

    verbs = (
        {
            d[0].lemma
            for d in dependencies
            if d[1] == RELATION_NOMINAL_SUBJECT
            and d[2].lemma in nouns
            and d[0].upos == "VERB"
        }
        | {
            d[2].lemma
            for d in dependencies
            if d[1] == RELATION_RELATIVE_CLAUSE_MODIFIER
            and d[0].lemma in nouns
            and d[2].upos == "VERB"
        }
        | {
            d[2].lemma
            for d in dependencies
            if d[1] == RELATION_ADJECTIVAL_CLAUSE
            and d[0].lemma in nouns
            and d[2].upos == "VERB"
        }
    )

    return verbs


def get_objects_for_noun(pos_tagged_caption, nouns):
    dependencies = pos_tagged_caption.dependencies

    objects = {
        d[2].lemma
        for d in dependencies
        if d[1] == RELATION_OBJECT
        or d[1] == RELATION_INDIRECT_OBJECT
        and d[0].lemma in nouns
    }
    return objects


def contains_adjective_noun_pair(pos_tagged_caption, nouns, adjectives):
    noun_is_present = False
    adjective_is_present = False

    for word in pos_tagged_caption.words:
        if word.lemma in nouns:
            noun_is_present = True
        if word.lemma in adjectives:
            adjective_is_present = True

    caption_adjectives = get_adjectives_for_noun(pos_tagged_caption, nouns)
    combination_is_present = bool(set(adjectives) & caption_adjectives)

    return noun_is_present, adjective_is_present, combination_is_present


def contains_verb_noun_pair(pos_tagged_caption, nouns, verbs):
    noun_is_present = False
    verb_is_present = False

    for word in pos_tagged_caption.words:
        if word.lemma in nouns:
            noun_is_present = True
        if word.lemma in verbs:
            verb_is_present = True

    caption_verbs = get_verbs_for_noun(pos_tagged_caption, nouns)
    combination_is_present = bool(set(verbs) & caption_verbs)

    return noun_is_present, verb_is_present, combination_is_present


def read_image(path):
    img = imread(path)
    if len(img.shape) == 2:  # b/w image
        img = img[:, :, np.newaxis]
        img = np.concatenate([img, img, img], axis=2)
    img = imresize(img, (256, 256))
    img = img.transpose(2, 0, 1)
    assert img.shape == (3, 256, 256)
    assert np.max(img) <= 255
    return img


def invert_normalization(image):
    image = torch.FloatTensor(image)
    inv_normalize = transforms.Normalize(
        mean=(-1 * np.array(IMAGENET_IMAGES_MEAN) / np.array(IMAGENET_IMAGES_STD)),
        std=(1 / np.array(IMAGENET_IMAGES_STD)),
    )
    return inv_normalize(image)


def get_splits_from_occurrences_data(heldout_pairs):
    occurrences_data_files = [
        os.path.join(base_dir, "data", "occurrences", pair + ".json")
        for pair in heldout_pairs
    ]
    test_images_split = set()
    val_images_split = set()

    for file in occurrences_data_files:
        with open(file, "r") as json_file:
            occurrences_data = json.load(json_file)

        test_images_split |= {
            key
            for key, value in occurrences_data[OCCURRENCE_DATA].items()
            if value[PAIR_OCCURENCES] >= 1 and value[DATA_COCO_SPLIT] == "val2014"
        }
        val_images_split |= {
            key
            for key, value in occurrences_data[OCCURRENCE_DATA].items()
            if value[PAIR_OCCURENCES] >= 1 and value[DATA_COCO_SPLIT] == "train2014"
        }

    with open(occurrences_data_files[0], "r") as json_file:
        occurrences_data = json.load(json_file)

    train_images_split = {
        key
        for key, value in occurrences_data[OCCURRENCE_DATA].items()
        if key not in val_images_split and value[DATA_COCO_SPLIT] == "train2014"
    }

    return list(train_images_split), list(val_images_split), list(test_images_split)


def show_img(img):
    plt.imshow(img.transpose(1, 2, 0))
    plt.axis("off")
    plt.show()


def decode_caption(encoded_caption, word_map):
    rev_word_map = {v: k for k, v in word_map.items()}
    return [rev_word_map[ind] for ind in encoded_caption]


def get_caption_without_special_tokens(caption, word_map):
    """Remove start, end and padding tokens from and encoded caption."""

    return [
        token
        for token in caption
        if token
        not in {word_map[TOKEN_START], word_map[TOKEN_END], word_map[TOKEN_PADDING]}
    ]


def clip_gradients(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def get_eval_log_file_path(checkpoint, dataset_splits, logging_dir="logs/eval"):
    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)
    checkpoint_name = os.path.basename(checkpoint).split(".")[0]
    return os.path.join(
        logging_dir, get_file_name_prefix(checkpoint_name, dataset_splits, "") + ".log"
    )


def get_train_log_file_path(
    model_name, dataset_splits, name_suffix, embeddings_file, logging_dir="logs/train"
):
    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)
    if embeddings_file:
        name_suffix += "_" + os.path.basename(embeddings_file).split(".")[0]
    return os.path.join(
        logging_dir,
        get_file_name_prefix(model_name, dataset_splits, name_suffix) + ".log",
    )


def get_file_name_prefix(model_name, dataset_splits, name_suffix):
    split_name = os.path.basename(dataset_splits).split(".")[0]

    return model_name + "_" + split_name + name_suffix


def get_checkpoint_file_path(
    model_name, dataset_splits, name_suffix, is_best, checkpoints_dir="checkpoints"
):
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
    name = (
        "checkpoint_"
        + get_file_name_prefix(model_name, dataset_splits, name_suffix)
        + ".pth.tar"
    )
    if is_best:
        name = "best_" + name

    return os.path.join(checkpoints_dir, name)


def save_checkpoint(
    model_name,
    dataset_splits,
    epoch,
    epochs_since_last_improvement,
    encoder,
    decoder,
    encoder_optimizer,
    decoder_optimizer,
    generation_metric_score,
    recall_metric_score,
    is_best,
    name_suffix,
):
    """
    Save a model checkpoint.

    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement
    :param encoder: encoder model
    :param decoder: decoder model
    :param encoder_optimizer: optimizer to update the encoder's weights
    :param decoder_optimizer: optimizer to update the decoder's weights
    :param validation_metric_score: validation set score for this epoch
    :param is_best: True, if this is the best checkpoint so far (will save the model to a dedicated file)
    """
    state = {
        "model_name": model_name,
        "epoch": epoch,
        "epochs_since_improvement": epochs_since_last_improvement,
        "generation_metric_score": generation_metric_score,
        "recall_metric_score": recall_metric_score,
        "encoder": encoder,
        "decoder": decoder,
        "encoder_optimizer": encoder_optimizer,
        "decoder_optimizer": decoder_optimizer,
    }
    file_name = get_checkpoint_file_path(model_name, dataset_splits, name_suffix, False)
    torch.save(state, file_name)

    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        file_name = get_checkpoint_file_path(
            model_name, dataset_splits, name_suffix, True
        )
        torch.save(state, file_name)


class AverageMeter(object):
    """Class to keep track of most recent, average, sum, and count of a metric."""

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
    Shrink the learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate should be shrunk.
    :param shrink_factor: factor to multiply learning rate with.
    """

    logging.info("\nAdjusting learning rate.")
    for param_group in optimizer.param_groups:
        param_group["lr"] = param_group["lr"] * shrink_factor
    logging.info(
        "The new learning rate is {}\n".format(optimizer.param_groups[0]["lr"])
    )


def load_embeddings(emb_file, word_map):
    """Return an embedding for the specified word map from the a GloVe embedding file"""

    logging.info("\nLoading embeddings: {}".format(emb_file))
    with open(emb_file, "r") as f:
        emb_dim = len(f.readline().split(" ")) - 1

    vocab = set(word_map.keys())

    embeddings = torch.FloatTensor(len(vocab), emb_dim)
    # Initialize the weights with random values (these will stay only for cases where a word in the vocabulary does not
    # exist in the loaded embeddings' vocabulary
    torch.nn.init.normal_(embeddings, 0, 1)

    tokens_found = set()
    for line in tqdm(open(emb_file, "r")):
        line_split = line.split(" ")
        emb_word = line_split[0]
        if emb_word in vocab:
            embedding = [float(t) for t in line_split[1:] if not t.isspace()]
            embeddings[word_map[emb_word]] = torch.FloatTensor(embedding)
            tokens_found.add(emb_word)

    missed_tokens = vocab - tokens_found
    if missed_tokens:
        logging.info(
            "\nThe loaded embeddings did not contain an embedding for the following tokens: {}".format(
                missed_tokens
            )
        )

    return embeddings, emb_dim
