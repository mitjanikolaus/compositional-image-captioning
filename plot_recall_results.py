import json

import numpy as np
import matplotlib.pyplot as plt

from metrics import average_recall

MODE_VERBOSE = "VERBOSE"
MODE_AVERAGE_FOR_PAIR = "AVERAGE_FOR_PAIRS"
MODE_AVERAGE_ALL = "AVERAGE_ALL"

eval_files = [
    "../eval/eval_checkpoint_BOTTOM_UP_TOP_DOWN_heldout_pairs_none_best.json",
    "../eval/eval_checkpoint_BOTTOM_UP_TOP_DOWN_heldout_pairs_glove_best.json",
    "../eval/eval_checkpoint_BOTTOM_UP_TOP_DOWN_RANKING_heldout_pairs_continued_joint_best.json",
]
mode = MODE_AVERAGE_ALL
eval_datas = []

for file in eval_files:
    with open(file, "r") as json_file:
        eval_data = json.load(json_file)

    eval_datas.append(eval_data)

if mode == MODE_AVERAGE_ALL:

    for j, eval_data in enumerate(eval_datas):
        true_positives = 0
        numbers = 0

        recall = average_recall(eval_data)
        print(recall)


else:
    fig, axes = plt.subplots(nrows=len(eval_datas[0]), sharex=True)
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
                rects1 = axis.bar(index + j * bar_width, recall, bar_width)

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

                rects1 = axis.bar(j * bar_width, recall, bar_width)

                axis.text(x=j * bar_width, y=0.8, s=np.round(recall, 2), size=5)
            axis.set_ylim(0, 1)
            axis.set_title(pair)
        plt.xticks([], [])

    # Common ylabel
    fig.text(0.06, 0.5, "Recall", ha="center", va="center", rotation="vertical")
    # Common legend
    fig.legend(
        axes,
        labels=["upper bound", "glove embeddings", "joint objective"],
        loc="upper right",
        borderaxespad=0.1,
    )

    plt.subplots_adjust(hspace=0.5)

    plt.show()
