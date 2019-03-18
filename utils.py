import json

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
BOTTOM_UP_FEATURES_FILENAME = "bottom_up_features.hdf5"
CAPTIONS_FILENAME = "captions.json"
CAPTION_LENGTHS_FILENAME = "caption_lengths.json"

NOUNS = "nouns"
ADJECTIVES = "adjectives"

OCCURRENCE_DATA = "adjective_noun_occurrence_data"
PAIR_OCCURENCES = "pair_occurrences"
NOUN_OCCURRENCES = "noun_occurrences"
ADJECTIVE_OCCURRENCES = "adjective_occurrences"

RELATION_NOMINAL_SUBJECT = "nsubj"
RELATION_ADJECTIVAL_MODIFIER = "amod"
RELATION_CONJUNCT = "conj"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def update_params(defaults, params):
    updated = defaults.copy()
    for key, value in defaults.items():
        if key in params and params[key]:
            updated[key] = params[key]
    return updated


def contains_adjective_noun_pair(nlp_pipeline, caption, nouns, adjectives):
    noun_is_present = False
    adjective_is_present = False

    doc = nlp_pipeline(caption)
    sentence = doc.sentences[0]

    for token in sentence.tokens:
        if token.text in nouns:
            noun_is_present = True
        if token.text in adjectives:
            adjective_is_present = True

    dependencies = sentence.dependencies
    caption_adjectives = {
        d[2].text
        for d in dependencies
        if d[1] == RELATION_ADJECTIVAL_MODIFIER and d[0].text in nouns
    } | {
        d[0].text
        for d in dependencies
        if d[1] == RELATION_NOMINAL_SUBJECT and d[2].text in nouns
    }
    conjuncted_caption_adjectives = set()
    for adjective in caption_adjectives:
        conjuncted_caption_adjectives.update(
            {
                d[2].text
                for d in dependencies
                if d[1] == RELATION_CONJUNCT and d[0].text == adjective
            }
            | {
                d[2].text
                for d in dependencies
                if d[1] == RELATION_ADJECTIVAL_MODIFIER and d[0].text == adjective
            }
        )

    caption_adjectives |= conjuncted_caption_adjectives
    combination_is_present = bool(adjectives & caption_adjectives)

    return noun_is_present, adjective_is_present, combination_is_present


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


def get_splits_from_occurrences_data(occurrences_data_file, val_set_size=0):
    with open(occurrences_data_file, "r") as json_file:
        occurrences_data = json.load(json_file)

    test_images_split = [
        key
        for key, value in occurrences_data[OCCURRENCE_DATA].items()
        if value[PAIR_OCCURENCES] >= 1
    ]

    indices_without_test = [
        key
        for key, value in occurrences_data[OCCURRENCE_DATA].items()
        if value[PAIR_OCCURENCES] == 0
    ]

    train_val_split = int((1 - val_set_size) * len(indices_without_test))
    train_images_split = indices_without_test[:train_val_split]
    val_images_split = indices_without_test[train_val_split:]

    return train_images_split, val_images_split, test_images_split


def show_img(img):
    plt.imshow(img.transpose(1, 2, 0))
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


def save_checkpoint(
    model_name,
    name,
    epoch,
    epochs_since_last_improvement,
    encoder,
    decoder,
    encoder_optimizer,
    decoder_optimizer,
    bleu4,
    is_best,
):
    """
    Save a model checkpoint.

    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement
    :param encoder: encoder model
    :param decoder: decoder model
    :param encoder_optimizer: optimizer to update the encoder's weights
    :param decoder_optimizer: optimizer to update the decoder's weights
    :param bleu4: validation set BLEU-4 score for this epoch
    :param is_best: True, if this is the best checkpoint so far (will save the model to a dedicated file)
    """
    state = {
        "model_name": model_name,
        "epoch": epoch,
        "epochs_since_improvement": epochs_since_last_improvement,
        "bleu-4": bleu4,
        "encoder": encoder,
        "decoder": decoder,
        "encoder_optimizer": encoder_optimizer,
        "decoder_optimizer": decoder_optimizer,
    }
    torch.save(state, "checkpoint_" + model_name + "_" + name + ".pth.tar")

    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, "checkpoint_" + model_name + "_" + name + "_best.pth.tar")


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

    print("\nAdjusting learning rate.")
    for param_group in optimizer.param_groups:
        param_group["lr"] = param_group["lr"] * shrink_factor
    print("The new learning rate is {}\n".format(optimizer.param_groups[0]["lr"]))


def top_k_accuracy(scores, targets, k):
    """
    Compute the top-k accuracy from predicted and true labels.

    :param scores: predicted scores from the model
    :param targets: true labels
    :param k: k
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()
    return correct_total.item() * (100.0 / batch_size)


def load_embeddings(emb_file, word_map):
    """Return an embedding for the specified word map from the a GloVe embedding file"""

    print("\nLoading embeddings: {}".format(emb_file))
    with open(emb_file, "r") as f:
        emb_dim = len(f.readline().split(" ")) - 1

    vocab = set(word_map.keys())

    embeddings = torch.FloatTensor(len(vocab), emb_dim, device=device)
    # Initialize the weights with random values (these will stay only for cases where a word in the vocabulary does not
    # exist in the loaded embeddings' vocabulary
    torch.nn.init.normal_(embeddings, 0, 1)

    tokens_found = set()
    for line in tqdm(open(emb_file, "r")):
        line_split = line.split(" ")
        emb_word = line_split[0]
        if emb_word in vocab:
            embedding = [float(t) for t in line_split[1:] if not t.isspace()]
            embeddings[word_map[emb_word]] = torch.FloatTensor(embedding, device=device)
            tokens_found.add(emb_word)

    missed_tokens = vocab - tokens_found
    if missed_tokens:
        print(
            "\nThe loaded embeddings did not contain an embedding for the following tokens: {}".format(
                missed_tokens
            )
        )

    return embeddings, emb_dim
