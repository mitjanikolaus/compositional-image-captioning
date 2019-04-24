import os
import sys

import argparse
import json

import numpy as np
import matplotlib.pyplot as plt

from metrics import average_recall

MODE_VERBOSE = "VERBOSE"
MODE_AVERAGE_FOR_PAIR = "AVERAGE_FOR_PAIRS"


def plot_recall_results(eval_files, mode, labels):
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
        label + " (Avg recall: {})".format(np.round(average_recall(eval_datas[i]), 2))
        for i, label in enumerate(labels)
    ]

    fig, axes = plt.subplots(nrows=len(eval_datas[0]), sharex=True, figsize=(8, 15))
    plt.suptitle("Recall")
    bar_width = 0.2
    if mode == MODE_VERBOSE:
        index = np.arange(5)

        for i, pair in enumerate(eval_datas[0].keys()):
            axis = axes[i]

            for j, eval_data in enumerate(eval_datas):
                recall = np.array(
                    list(eval_data[pair]["true_positives"].values())
                ) / np.array(list(eval_data[pair]["numbers"].values()))
                axis.bar(index + j * bar_width, recall, bar_width)

            axis.set_ylim(0, 1)
            axis.set_title(pair)

        plt.xticks(index + bar_width, index + 1)
        plt.xlabel("Agreement within reference captions")

    elif mode == MODE_AVERAGE_FOR_PAIR:
        for i, pair in enumerate(eval_datas[0].keys()):
            axis = axes[i]
            for j, eval_data in enumerate(eval_datas):
                recall = np.sum(
                    list(eval_data[pair]["true_positives"].values())
                ) / np.sum(list(eval_data[pair]["numbers"].values()))

                axis.bar(j * bar_width, recall, bar_width)

                axis.text(x=j * bar_width, y=0.8, s=np.round(recall, 2), size=7)
            axis.set_ylim(0, 1)
            axis.set_title(pair)
        plt.xticks([], [])

    # Common ylabel
    fig.text(0.06, 0.5, "Recall", ha="center", va="center", rotation="vertical")
    # Common legend
    fig.legend(labels=labels, loc="lower center", borderaxespad=0.1)

    plt.subplots_adjust(hspace=0.5)

    plt.show()


def check_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--eval-files", nargs="+", help="Files containing evaluation statistics"
    )
    parser.add_argument(
        "--mode",
        help="Mode",
        default=MODE_AVERAGE_FOR_PAIR,
        choices=[MODE_AVERAGE_FOR_PAIR, MODE_VERBOSE],
    )
    parser.add_argument(
        "--labels", nargs="+", help="Labels for each model that was evaluated"
    )
    parsed_args = parser.parse_args(args)
    print(parsed_args)
    return parsed_args


if __name__ == "__main__":
    parsed_args = check_args(sys.argv[1:])
    plot_recall_results(parsed_args.eval_files, parsed_args.mode, parsed_args.labels)
