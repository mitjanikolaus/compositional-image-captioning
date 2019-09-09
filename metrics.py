"""Metrics for the image captioning task"""

import json
import logging
import os
from collections import Counter

import matplotlib.pyplot as plt

import stanfordnlp

import numpy as np
from utils import (
    decode_caption,
    NOUNS,
    ADJECTIVES,
    contains_adjective_noun_pair,
    OCCURRENCE_DATA,
    PAIR_OCCURENCES,
    get_caption_without_special_tokens,
    VERBS,
    contains_verb_noun_pair,
    get_splits_from_occurrences_data,
    get_adjectives_for_noun,
    get_verbs_for_noun,
)

# stanfordnlp.download('en', confirm_if_exists=True)

base_dir = os.path.dirname(os.path.abspath(__file__))


def recall_pairs(generated_captions, word_map, heldout_pairs, output_file_name):
    logging.info("\n\nRecall@{}:".format(len(next(iter(generated_captions.values())))))
    recall_scores = {}
    nlp_pipeline = stanfordnlp.Pipeline()
    for pair in heldout_pairs:
        occurrences_data_file = os.path.join(
            base_dir, "data", "occurrences", pair + ".json"
        )
        occurrences_data = json.load(open(occurrences_data_file, "r"))

        _, _, test_indices = get_splits_from_occurrences_data([pair])
        nouns = set(occurrences_data[NOUNS])

        if ADJECTIVES in occurrences_data:
            adjectives = set(occurrences_data[ADJECTIVES])
            recall_score = calc_recall(
                generated_captions,
                test_indices,
                word_map,
                nouns,
                adjectives,
                occurrences_data,
                contains_adjective_noun_pair,
                nlp_pipeline,
            )
        elif VERBS in occurrences_data:
            verbs = set(occurrences_data[VERBS])
            recall_score = calc_recall(
                generated_captions,
                test_indices,
                word_map,
                nouns,
                verbs,
                occurrences_data,
                contains_verb_noun_pair,
                nlp_pipeline,
            )
        else:
            raise ValueError("No adjectives or verbs found in occurrences data!")

        pair = os.path.basename(occurrences_data_file).split(".")[0]
        recall_scores[pair] = recall_score
        average_pair_recall = np.sum(
            list(recall_score["true_positives"].values())
        ) / np.sum(list(recall_score["numbers"].values()))
        logging.info("{}: {}".format(pair, np.round(average_pair_recall, 2)))

    logging.info("Average: {}".format(average_recall(recall_scores)))
    json.dump(recall_scores, open(output_file_name, "w"))


def calc_recall(
    generated_captions,
    test_indices,
    word_map,
    nouns,
    other,
    occurrences_data,
    contains_pair_function,
    nlp_pipeline,
):
    true_positives = dict.fromkeys(["N=1", "N=2", "N=3", "N=4", "N=5"], 0)
    numbers = dict.fromkeys(["N=1", "N=2", "N=3", "N=4", "N=5"], 0)
    adjective_frequencies = Counter()
    verb_frequencies = Counter()
    for coco_id in test_indices:
        top_k_captions = generated_captions[coco_id]
        count = occurrences_data[OCCURRENCE_DATA][coco_id][PAIR_OCCURENCES]

        hit = False
        for caption in top_k_captions:
            caption = " ".join(
                decode_caption(
                    get_caption_without_special_tokens(caption, word_map), word_map
                )
            )
            pos_tagged_caption = nlp_pipeline(caption).sentences[0]
            _, _, contains_pair = contains_pair_function(
                pos_tagged_caption, nouns, other
            )
            if contains_pair:
                hit = True

            noun_is_present = False
            for word in pos_tagged_caption.words:
                if word.lemma in nouns:
                    noun_is_present = True
            if noun_is_present:
                adjectives = get_adjectives_for_noun(pos_tagged_caption, nouns)
                if len(adjectives) == 0:
                    adjective_frequencies["No adjective"] += 1
                adjective_frequencies.update(adjectives)

                verbs = get_verbs_for_noun(pos_tagged_caption, nouns)
                if len(verbs) == 0:
                    verb_frequencies["No verb"] += 1
                verb_frequencies.update(verbs)

        if hit:
            true_positives["N={}".format(count)] += 1
        numbers["N={}".format(count)] += 1

    recall_score = {
        "true_positives": true_positives,
        "numbers": numbers,
        "adjective_frequencies": adjective_frequencies,
        "verb_frequencies": verb_frequencies,
    }
    return recall_score


def average_recall(recall_scores, min_importance=1):
    pair_recalls_summed = 0
    length = 0

    for i, pair in enumerate(recall_scores.keys()):
        average_pair_recall = np.sum(
            list(recall_scores[pair]["true_positives"].values())[min_importance - 1 :]
        ) / np.sum(list(recall_scores[pair]["numbers"].values())[min_importance - 1 :])
        if not np.isnan(average_pair_recall):
            pair_recalls_summed += average_pair_recall
            length += 1

    recall = pair_recalls_summed / length
    return recall


def beam_occurrences(
    generated_beams, beam_size, word_map, heldout_pairs, max_print_length=20
):
    for pair in heldout_pairs:
        occurrences_data_file = os.path.join(
            base_dir, "data", "occurrences", pair + ".json"
        )
        occurrences_data = json.load(open(occurrences_data_file, "r"))

        _, _, test_indices = get_splits_from_occurrences_data([pair])

        nouns = set(occurrences_data[NOUNS])
        if ADJECTIVES in occurrences_data:
            adjectives = set(occurrences_data[ADJECTIVES])
        if VERBS in occurrences_data:
            verbs = set(occurrences_data[VERBS])

        max_length = max([beams[-1].size(1) for beams in generated_beams.values()])
        noun_occurrences = np.zeros(max_length)
        other_occurrences = np.zeros(max_length)
        pair_occurrences = np.zeros(max_length)

        num_beams = np.zeros(max_length)

        for coco_id in test_indices:
            beam = generated_beams[coco_id]
            for step, beam_timestep in enumerate(beam):
                noun_match = False
                other_match = False
                pair_match = False
                for branch in beam_timestep:
                    branch_words = set(decode_caption(branch.numpy(), word_map))
                    noun_occurs = bool(nouns & branch_words)

                    if ADJECTIVES in occurrences_data:
                        adjective_occurs = bool(adjectives & branch_words)
                        if adjective_occurs:
                            other_match = True
                    elif VERBS in occurrences_data:
                        verb_occurs = bool(verbs & branch_words)
                        if verb_occurs:
                            other_match = True
                    if noun_occurs:
                        noun_match = True

                    if noun_occurs and other_match:
                        pair_match = True
                if noun_match:
                    noun_occurrences[step] += 1
                if other_match:
                    other_occurrences[step] += 1
                if pair_match:
                    pair_occurrences[step] += 1
                num_beams[step] += 1

        # Print only occurrences up to max_print_length
        print_length = min(max_print_length, len(np.trim_zeros(num_beams)))

        name = os.path.basename(occurrences_data_file).split(".")[0]
        logging.info("Beam occurrences for {}".format(name))
        logging.info("Nouns: {}".format(noun_occurrences[:print_length]))
        logging.info("Adjectives/Verbs: {}".format(other_occurrences[:print_length]))
        logging.info("Pairs: {}".format(pair_occurrences[:print_length]))
        logging.info("Number of beams: {}".format(num_beams[:print_length]))

        steps = np.arange(print_length)

        plt.plot(
            steps,
            noun_occurrences[:print_length] / num_beams[:print_length],
            label="nouns",
        )
        plt.plot(
            steps,
            other_occurrences[:print_length] / num_beams[:print_length],
            label="adjectives/verbs",
        )
        plt.plot(
            steps,
            pair_occurrences[:print_length] / num_beams[:print_length],
            label="pairs",
        )
        plt.legend()
        plt.xlabel("timestep")
        plt.title("Recall@{} for {} in the decoding beam".format(beam_size, name))
        plt.show()
