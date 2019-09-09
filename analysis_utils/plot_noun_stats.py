"""Plot the stats calculated using noun_stats.py"""

import sys
from collections import Counter

import argparse
import json

import matplotlib.pyplot as plt


def plot_noun_stats_results(noun_stats_file):
    with open(noun_stats_file, "r") as json_file:
        noun_stats = json.load(json_file)

    for noun, stats in noun_stats.items():

        adjective_frequencies = Counter(stats["adjective_frequencies"])
        verb_frequencies = Counter(stats["verb_frequencies"])

        total = sum(adjective_frequencies.values())
        no_adjective_freq = int(adjective_frequencies["No adjective"] / total * 100)
        del adjective_frequencies["No adjective"]

        total = sum(verb_frequencies.values())
        no_verb_freq = int(verb_frequencies["No verb"] / total * 100)
        del verb_frequencies["No verb"]

        fig, axes = plt.subplots(nrows=2, figsize=(30, 15))
        plt.suptitle("{}".format(noun) + " ({} captions)".format(total))

        axes[0].bar(
            [adj for adj, freq in adjective_frequencies.most_common(20)],
            [freq for adj, freq in adjective_frequencies.most_common(20)],
        )
        axes[0].set_title(
            "Adjectives (captions w/o adjective: {}%)".format(no_adjective_freq)
        )
        axes[1].bar(
            [adj for adj, freq in verb_frequencies.most_common(20)],
            [freq for adj, freq in verb_frequencies.most_common(20)],
        )
        axes[1].set_title("Verbs (captions w/o verb: {}%)".format(no_verb_freq))

        # plt.bar
        plt.show()


def check_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument("--noun-stats", help="File containing noun stats")
    parsed_args = parser.parse_args(args)
    print(parsed_args)
    return parsed_args


if __name__ == "__main__":
    parsed_args = check_args(sys.argv[1:])
    plot_noun_stats_results(parsed_args.noun_stats)
