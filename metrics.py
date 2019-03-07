import json

import numpy as np
from utils import decode_caption, NOUNS, ADJECTIVES, contains_adjective_noun_pair


def adjective_noun_matches(
    target_captions, generated_captions, occurrences_data, word_map
):
    import stanfordnlp

    # stanfordnlp.download('en', confirm_if_exists=True)
    nlp_pipeline = stanfordnlp.Pipeline()

    nouns = set(occurrences_data[NOUNS])
    adjectives = set(occurrences_data[ADJECTIVES])
    # nouns = {'machine', 'auto', 'car', 'motorcar', 'automobile'}
    # adjectives = {'white'}

    true_positives = np.zeros(5)
    false_negatives = np.zeros(5)
    for i, caption in enumerate(generated_captions):
        count = 0
        for target_caption in target_captions[i]:
            target_caption = " ".join(decode_caption(target_caption, word_map))
            _, _, target_match = contains_adjective_noun_pair(
                nlp_pipeline, target_caption, nouns, adjectives
            )
            if target_match:
                count += 1

        caption = " ".join(decode_caption(caption, word_map))
        _, _, match = contains_adjective_noun_pair(
            nlp_pipeline, caption, nouns, adjectives
        )
        for j in range(count):
            if match:
                true_positives[j] += 1
            else:
                false_negatives[j] += 1

    print(true_positives)
    print(false_negatives)
    recall = true_positives / (true_positives + false_negatives)
    print(recall)
    return recall
