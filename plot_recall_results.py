import os
import sys

import argparse
import json

import numpy as np
import matplotlib.pyplot as plt

from metrics import average_recall

# Plot verbose results
MODE_VERBOSE = "VERBOSE"
# Plot only one average recall per pair
MODE_AVERAGE_FOR_PAIR = "AVG_PAIRS"
# Plot only one average recall per category
MODE_AVERAGE_FOR_CATEGORY = "AVG_CATEGORIES"

ADJECTIVES_COLORS_ANIMATE = {"black_cat", "brown_dog", "white_horse", "black_bird"}
ADJECTIVES_COLORS_INANIMATE = {"red_bus", "white_truck", "blue_bus", "white_boat"}

ADJECTIVES_SIZES_ANIMATE = {"big_bird", "small_cat", "big_cat", "small_dog"}
ADJECTIVES_SIZES_INANIMATE = {"small_plane", "big_plane", "small_table", "big_truck"}

VERBS_TRANSITIVE = {"eat_man", "ride_woman", "hold_child", "eat_horse"}
VERBS_INTRANSITIVE = {"lie_woman", "fly_bird", "stand_bird", "stand_child"}

CATEGORIES = {
    "colors animate": ADJECTIVES_COLORS_ANIMATE,
    "colors inanimate": ADJECTIVES_COLORS_INANIMATE,
    "sizes animate": ADJECTIVES_SIZES_ANIMATE,
    "sizes inanimate": ADJECTIVES_SIZES_INANIMATE,
    "verbs transitive": VERBS_TRANSITIVE,
    "verbs intransitive": VERBS_INTRANSITIVE,
}


def calc_average_for_pair(stats, min_importance):
    return np.sum(
        list(stats["true_positives"].values())[min_importance - 1 :]
    ) / np.sum(list(stats["numbers"].values())[min_importance - 1 :])


def plot_recall_results(eval_files, mode, labels, min_importance):
    if not labels:
        labels = [
            os.path.basename(path).split("checkpoint_")[-1] for path in eval_files
        ]

    eval_datas = []

    for file in eval_files:
        with open(file, "r") as json_file:
            eval_data = json.load(json_file)

        eval_datas.append(eval_data)

    labels = [
        label
        + " (Avg recall: {})".format(
            np.round(average_recall(eval_datas[i], min_importance), 3)
        )
        for i, label in enumerate(labels)
    ]

    bar_width = 0.1
    if mode == MODE_VERBOSE:
        fig, axes = plt.subplots(nrows=len(eval_datas[0]), sharex=True, figsize=(8, 15))

        index = np.arange(5 - (min_importance - 1))

        for i, pair in enumerate(eval_datas[0].keys()):
            axis = axes[i]

            for j, eval_data in enumerate(eval_datas):
                recall = np.array(
                    list(eval_data[pair]["true_positives"].values())[
                        min_importance - 1 :
                    ]
                ) / np.array(
                    list(eval_data[pair]["numbers"].values())[min_importance - 1 :]
                )
                axis.bar(index + j * bar_width, recall, bar_width)

            axis.set_ylim(0, 1)
            axis.set_title(pair)

        plt.xticks(index + bar_width, index + min_importance)
        plt.xlabel("Importance (Agreement within reference captions)")

    elif mode == MODE_AVERAGE_FOR_PAIR:
        fig, axes = plt.subplots(nrows=len(eval_datas[0]), sharex=True, figsize=(8, 15))

        for i, pair in enumerate(eval_datas[0].keys()):
            axis = axes[i]
            for j, eval_data in enumerate(eval_datas):
                recall = calc_average_for_pair(eval_data[pair], min_importance)

                axis.bar(j * bar_width, recall, bar_width)

                axis.text(x=j * bar_width, y=0.8, s=np.round(recall, 3), size=7)
            axis.set_ylim(0, 1)
            axis.set_title(pair)
        plt.xticks([], [])

    elif mode == MODE_AVERAGE_FOR_CATEGORY:
        category_eval_datas = []

        for eval_data in eval_datas:
            category_eval_data = {}
            for category_name, category in CATEGORIES.items():
                average = 0
                num_pairs = 0
                for pair in category:
                    average += calc_average_for_pair(eval_data[pair], min_importance)
                    num_pairs += 1
                category_eval_data[category_name] = average / num_pairs
            category_eval_datas.append(category_eval_data)

        fig, axes = plt.subplots(
            nrows=len(category_eval_datas[0]), sharex=True, figsize=(8, 15)
        )

        for category_name in category_eval_datas[0].keys():
            print(category_name, end=" ")
        for i, eval_data in enumerate(category_eval_datas):
            print("\n" + labels[i], end=" ")
            for value in eval_data.values():
                print(np.round(value, 3), end=" ")

        for i, category_name in enumerate(category_eval_datas[0].keys()):
            axis = axes[i]
            for j, category_eval_data in enumerate(category_eval_datas):
                recall = category_eval_data[category_name]

                axis.bar(j * bar_width, recall, bar_width)

                axis.text(x=j * bar_width, y=0.8, s=np.round(recall, 3), size=7)
            axis.set_ylim(0, 1)
            axis.set_title(category_name)
        plt.xticks([], [])

    plt.suptitle("Performance (min importance={})".format(min_importance))
    # Common ylabel
    fig.text(0.06, 0.5, "Recall", ha="center", va="center", rotation="vertical")
    # Common legend
    fig.legend(
        labels=labels, loc="lower center", borderaxespad=0.1, bbox_to_anchor=(0.5, 0)
    )

    plt.subplots_adjust(hspace=0.5, bottom=0.18)

    plt.show()


def check_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--eval-files", nargs="+", help="Files containing evaluation statistics"
    )
    parser.add_argument(
        "--mode",
        help="Mode",
        default=MODE_AVERAGE_FOR_CATEGORY,
        choices=[MODE_AVERAGE_FOR_PAIR, MODE_VERBOSE, MODE_AVERAGE_FOR_CATEGORY],
    )
    parser.add_argument(
        "--labels", nargs="+", help="Labels for each model that was evaluated"
    )
    parser.add_argument(
        "--min-importance",
        help="Minimum importance (agreement between the reference captions)",
        type=int,
        default=1,
    )
    parsed_args = parser.parse_args(args)
    print(parsed_args)
    return parsed_args


if __name__ == "__main__":
    parsed_args = check_args(sys.argv[1:])
    plot_recall_results(
        parsed_args.eval_files,
        parsed_args.mode,
        parsed_args.labels,
        parsed_args.min_importance,
    )
