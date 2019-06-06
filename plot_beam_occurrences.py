import numpy as np

import matplotlib.pyplot as plt


def plot_beam_occurrences():
    beam_size = 5
    max_print_length = 20

    #   name = "black cat"
    #   num_beams = [448, 448, 448, 448, 448, 448, 448, 448, 448, 447, 417, 326, 216, 100,
    # 43,  11,   6,   3,   2]
    #
    #
    #   noun_occurrences = [247, 415, 401, 394, 406, 401, 393, 393, 397, 396, 369, 284, 194,  88,
    # 35,  10,   5,   2,   2]
    #   other_occurrences = [ 17, 166,  97,  91,  85,  81,  76,  79,  79,  77,  64,  48,  24,  15,
    # 11,   3,   1,   1,   0]
    #
    #   pair_occurrences = [ 2, 64, 66, 68, 70, 66, 60, 60, 61, 60, 45, 26, 14,  7,  5,  2,  0,  0,
    # 0]

    # name = "brown dog"
    # noun_occurrences = [179, 256, 244, 240, 251, 249, 240, 243, 252, 248, 234, 194, 127,  55,
    #  24,  10,   3,   1,   1,]
    # other_occurrences = [1, 51, 25, 25, 22, 20, 22, 19, 18, 17, 11,  7,  1,  0,  0,  0,  0,  0,
    #                                 0,]
    #
    # pair_occurrences = [0, 12, 15, 15, 15, 10, 10, 11, 11, 10,  6,  5,  1,  0,  0,  0,  0,  0,
    #                    0,]
    #
    # num_beams = [291, 291, 291, 291, 291, 291, 291, 291, 291, 291, 276, 228, 152,  63,
    #              27,  12,   3,   1,   1,]

    # name = "white truck"
    # noun_occurrences = [0, 81, 79, 74, 73, 77, 80, 83, 83, 83, 76, 64, 26, 13,  8,  6,  4,  1,]
    # other_occurrences = [3, 48, 33, 35, 32, 28, 30, 30, 32, 27, 26, 20, 15, 11,  7,  2,  1,  0,]
    #
    # pair_occurrences = [0, 13, 26, 23, 21, 19, 17, 17, 18, 17, 12,  9,  7,  3,  3,  1,  1,  0,]
    #
    # num_beams = [121, 121, 121, 121, 121, 121, 121, 121, 121, 120, 117, 102,  51,  29,
    #         17,   9,   5,   2,]

    # name = "blue bus"
    # noun_occurrences = [48, 128, 120, 124, 121, 119, 124, 124, 131, 128, 117, 102,  53,  20,
    #          11,   8,   3,]
    # other_occurrences = [5, 24, 16, 15, 13, 10,  9,  7,  7,  7,  5,  4,  3,  2,  0,  0,  0,]
    # pair_occurrences = [0,  6,  9,  9, 11,  7,  6,  6,  6,  6,  4,  3,  2,  2,  0,  0,  0,]
    #
    # num_beams = [143, 143, 143, 143, 143, 143, 143, 143, 143, 139, 133, 116,  63,  24,
    #         11,   8,   3,]

    # name = "blue bus" # butd + G
    # noun_occurrences = [37, 131, 125, 127, 123, 122, 127, 127, 136, 128, 110,  84,  23,   6,]
    # other_occurrences = [3, 26, 17, 14, 13, 12, 12, 10, 10, 11,  8,  6,  1,  1,]
    # pair_occurrences = [0,  6, 15, 13, 12, 11, 10,  9,  9, 10,  8,  5,  0,  0,]
    #
    # num_beams = [143, 143, 143, 143, 143, 143, 143, 143, 143, 137, 119,  94,  31,   9,]
    #

    name = "blue bus"  # full data
    noun_occurrences = [
        51,
        129,
        126,
        130,
        129,
        129,
        127,
        133,
        136,
        136,
        122,
        107,
        42,
        16,
        3,
        2,
        1,
        1,
    ]
    other_occurrences = [
        43,
        84,
        69,
        76,
        73,
        68,
        63,
        63,
        65,
        60,
        48,
        37,
        18,
        9,
        2,
        2,
        1,
        1,
    ]
    pair_occurrences = [
        15,
        34,
        57,
        70,
        70,
        66,
        59,
        57,
        57,
        57,
        45,
        35,
        17,
        8,
        1,
        1,
        1,
        1,
    ]

    num_beams = [
        143,
        143,
        143,
        143,
        143,
        143,
        143,
        143,
        143,
        143,
        132,
        116,
        47,
        19,
        4,
        3,
        1,
        1,
    ]

    print_length = min(max_print_length, len(np.trim_zeros(num_beams)))
    steps = np.arange(print_length)

    plt.plot(
        steps,
        np.array(noun_occurrences[:print_length]) / np.array(num_beams[:print_length]),
        label="noun",
    )
    plt.plot(
        steps,
        np.array(other_occurrences[:print_length]) / np.array(num_beams[:print_length]),
        label="adjective",
    )
    plt.plot(
        steps,
        np.array(pair_occurrences[:print_length]) / np.array(num_beams[:print_length]),
        label="pair",
    )
    plt.legend()
    plt.xlabel("timestep")
    plt.ylabel("recall@5")
    plt.title("Beam occurrences for {}".format(name))
    plt.show()


if __name__ == "__main__":
    plot_beam_occurrences()
