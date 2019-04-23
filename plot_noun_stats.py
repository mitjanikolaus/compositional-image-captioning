import sys
from collections import Counter

import argparse
import json

import matplotlib.pyplot as plt


def plot_noun_stats_results(noun_stats_file):
    with open(noun_stats_file, "r") as json_file:
        noun_stats = json.load(json_file)

    for noun, stats in noun_stats.items():

        stats = Counter(stats)

        total = sum(stats.values())
        no_adjective_freq = int(stats["No adjective"] / total * 100)
        del stats["No adjective"]
        top_stats = stats.most_common(20)

        plt.figure(figsize=(15, 8))
        plt.title(
            "Noun: {}".format(noun)
            + " ({} captions)".format(total)
            + " (captions w/o adjective: {}%)".format(no_adjective_freq)
        )
        plt.bar([adj for adj, freq in top_stats], [freq for adj, freq in top_stats])
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
