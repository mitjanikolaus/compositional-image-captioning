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
        logging.info(
            "Most common adjectives: {}".format(
                recall_score["adjective_frequencies"].most_common(10)
            )
        )
        logging.info(
            "Most common verbs: {}".format(
                recall_score["verb_frequencies"].most_common(10)
            )
        )

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

    recall_score = {}
    recall_score["true_positives"] = true_positives
    recall_score["numbers"] = numbers
    recall_score["adjective_frequencies"] = adjective_frequencies
    recall_score["verb_frequencies"] = verb_frequencies
    return recall_score


def average_recall(recall_scores, min_importance=1):
    pair_recalls_summed = 0

    for i, pair in enumerate(recall_scores.keys()):
        average_pair_recall = np.sum(
            list(recall_scores[pair]["true_positives"].values())[min_importance - 1 :]
        ) / np.sum(list(recall_scores[pair]["numbers"].values())[min_importance - 1 :])
        pair_recalls_summed += average_pair_recall

    recall = pair_recalls_summed / len(recall_scores)
    return recall


def accurracy_robust_coco(generated_captions, word_map):
    object_class_data_path = os.path.join(
        base_dir, "data", "robust_coco", "coco_obj_stats" + ".json"
    )
    object_class_data = json.load(open(object_class_data_path, "r"))
    object_class_map = {
        str(stat["image_id"]): stat["pclss"] for stat in object_class_data
    }

    test_pair_path = os.path.join(
        base_dir, "data", "robust_coco", "test_pair_list" + ".json"
    )
    test_pair_list = json.load(open(test_pair_path, "r"))
    test_pair_list = [pair[:2] for pair in test_pair_list]

    dict_path = os.path.join(base_dir, "data", "robust_coco", "dic_coco" + ".json")
    dict = json.load(open(dict_path, "r"))
    object_index_to_word = {index: word for word, index in dict["wtod"].items()}

    accurracies = []
    for coco_id, top_k_captions in generated_captions.items():
        caption = top_k_captions[0]
        caption = decode_caption(
            get_caption_without_special_tokens(caption, word_map), word_map
        )
        print(caption)

        # Get all object classes for the given image
        target_object_classes = object_class_map[coco_id]
        # Filter out only the compositionally novel classes:
        target_object_classes = set(
            np.array(
                [
                    pair
                    for pair in test_pair_list
                    if pair[0] in target_object_classes
                    and pair[1] in target_object_classes
                ]
            ).flatten()
        )
        print(
            "target classes: ",
            [object_index_to_word[index] for index in target_object_classes],
        )
        found_object_classes = set()
        for word in caption:
            if word in dict["wtol"]:
                lemma = dict["wtol"][word]
                if lemma in dict["wtod"]:
                    object_class = dict["wtod"][lemma]
                    if object_class in target_object_classes:
                        print("Found {}({})".format(lemma, object_class))
                        found_object_classes.add(object_class)
        accurracies.append(len(found_object_classes) / len(target_object_classes))

    accuracy = np.mean(accurracies)
    logging.info("Accuracy: {}".format(np.round(accuracy, 2)))


def beam_occurrences(
    generated_beams, beam_size, word_map, heldout_pairs, max_print_length=20
):
    for pair in heldout_pairs:
        occurrences_data_file = os.path.join(
            base_dir, "data", "occurrences", pair + ".json"
        )
        occurrences_data = json.load(open(occurrences_data_file, "r"))

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

        _, _, test_indices = get_splits_from_occurrences_data([[pair]])

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

        name = os.path.basename(occurrences_data_file).split(".")[0]
        logging.info("Beam occurrences for {}".format(name))
        logging.info("Nouns: {}".format(noun_occurrences))
        logging.info("Adjectives/Verbs: {}".format(other_occurrences))
        logging.info("Pairs: {}".format(pair_occurrences))
        logging.info("Number of beams: {}".format(num_beams))

        # Print only occurrences up to max_print_length
        print_length = min(max_print_length, len(np.trim_zeros(num_beams)))

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


def recall_captions_from_images(embedded_images, embedded_captions, testing_indices):
    embedding_size = next(iter(embedded_captions.values())).shape[1]
    all_captions = np.array(list(embedded_captions.values())).reshape(
        -1, embedding_size
    )
    all_captions_keys = list(embedded_captions.keys())

    index_list = []
    ranks = np.zeros(len(testing_indices))
    for i, key in enumerate(testing_indices):
        image = embedded_images[key]

        # Compute similarity of image to all captions
        d = np.dot(image, all_captions.T).flatten()
        inds = np.argsort(d)[::-1]
        index_list.append(inds[0])

        # Look for rank of all 5 corresponding captions
        best_rank = len(all_captions)
        index = all_captions_keys.index(key)
        for j in range(5 * index, 5 * index + 5, 1):
            rank = np.where(inds == j)[0]
            if rank < best_rank:
                best_rank = rank
        ranks[i] = best_rank

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    recalls_sum = r1 + r5 + r10
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    print("R@1: {}".format(r1))
    print("R@5: {}".format(r5))
    print("R@10: {}".format(r10))
    print("Sum of recalls: {}".format(recalls_sum))
    print("Median Rank: {}".format(medr))
    print("Mean Rank: {}".format(meanr))

    return recalls_sum


def recall_captions_from_images_pairs(
    embedded_images,
    embedded_captions,
    target_captions,
    word_map,
    occurrences_data_files,
):
    print("Recall@5 of pairs:")
    print(
        "Pair | Recall (n=1) | Recall (n=2) | Recall (n=3) | Recall (n=4) | Recall (n=5)"
    )
    nlp_pipeline = stanfordnlp.Pipeline()

    embedding_size = next(iter(embedded_captions.values())).shape[1]

    all_captions = np.array(list(embedded_captions.values())).reshape(
        -1, embedding_size
    )
    target_captions = np.array(list(target_captions.values())).reshape(
        len(all_captions), -1
    )

    for file in occurrences_data_files:
        with open(file, "r") as json_file:
            occurrences_data = json.load(json_file)

        _, evaluation_indices = get_ranking_splits_from_occurrences_data([file])

        name = os.path.basename(file).split(".")[0]

        nouns = set(occurrences_data[NOUNS])
        if ADJECTIVES in occurrences_data:
            adjectives = set(occurrences_data[ADJECTIVES])
        elif VERBS in occurrences_data:
            verbs = set(occurrences_data[VERBS])
        else:
            raise ValueError("No adjectives or verbs found in occurrences data!")

        index_list = []
        true_positives = np.zeros(5)
        false_negatives = np.zeros(5)
        for i, coco_id in enumerate(evaluation_indices):
            image = embedded_images[coco_id]

            # Compute similarity of image to all captions
            d = np.dot(image, all_captions.T).flatten()
            inds = np.argsort(d)[::-1]
            index_list.append(inds[0])

            count = occurrences_data[OCCURRENCE_DATA][coco_id][PAIR_OCCURENCES]

            # Look for pair occurrences in top 5 captions
            hit = False
            for j in inds[:5]:
                caption = " ".join(
                    decode_caption(
                        get_caption_without_special_tokens(
                            target_captions[j], word_map
                        ),
                        word_map,
                    )
                )
                pos_tagged_caption = nlp_pipeline(caption).sentences[0]
                contains_pair = False
                if ADJECTIVES in occurrences_data:
                    _, _, contains_pair = contains_adjective_noun_pair(
                        pos_tagged_caption, nouns, adjectives
                    )
                elif VERBS in occurrences_data:
                    _, _, contains_pair = contains_verb_noun_pair(
                        pos_tagged_caption, nouns, verbs
                    )
                if contains_pair:
                    hit = True

            if hit:
                true_positives[count - 1] += 1
            else:
                false_negatives[count - 1] += 1

        # Compute metrics
        recall = true_positives / (true_positives + false_negatives)

        print("\n" + name, end=" | ")
        for n in range(len(recall)):
            print(float("%.2f" % recall[n]), end=" | ")
