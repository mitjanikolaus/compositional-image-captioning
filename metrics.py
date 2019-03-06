import json

import numpy as np
from utils import decode_caption


RELATION_NOMINAL_SUBJECT = "nsubj"
RELATION_ADJECTIVAL_MODIFIER = "amod"
RELATION_CONJUNCT = "conj"


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


def adjective_noun_matches(target_captions, generated_captions, word_map):
    import stanfordnlp

    # stanfordnlp.download('en', confirm_if_exists=True)
    nlp_pipeline = stanfordnlp.Pipeline()

    with open("data/dogs.json", "r") as json_file:
        nouns = set(json.load(json_file))
    with open("data/brown.json", "r") as json_file:
        adjectives = set(json.load(json_file))

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
