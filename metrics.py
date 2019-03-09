import json

import numpy as np
from utils import (
    decode_caption,
    NOUNS,
    ADJECTIVES,
    contains_adjective_noun_pair,
    OCCURRENCE_DATA,
    PAIR_OCCURENCES,
    get_caption_without_special_tokens,
)


def recall_adjective_noun_pairs(
    generated_captions, coco_ids, word_map, occurrences_data_file
):
    import stanfordnlp

    # stanfordnlp.download('en', confirm_if_exists=True)
    nlp_pipeline = stanfordnlp.Pipeline()

    with open(occurrences_data_file, "r") as json_file:
        occurrences_data = json.load(json_file)

    nouns = set(occurrences_data[NOUNS])
    adjectives = set(occurrences_data[ADJECTIVES])

    true_positives = np.zeros(5)
    false_negatives = np.zeros(5)
    for coco_id, caption in zip(coco_ids, generated_captions):
        count = occurrences_data[OCCURRENCE_DATA][coco_id][PAIR_OCCURENCES]

        caption = " ".join(
            decode_caption(
                get_caption_without_special_tokens(caption, word_map), word_map
            )
        )
        _, _, match = contains_adjective_noun_pair(
            nlp_pipeline, caption, nouns, adjectives
        )
        for j in range(count):
            if match:
                true_positives[j] += 1
            else:
                false_negatives[j] += 1

    recall = true_positives / (true_positives + false_negatives)
    return recall
