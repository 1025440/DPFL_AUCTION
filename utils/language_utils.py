

import re
import numpy as np
import json


ALL_LETTERS = "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
NUM_LETTERS = len(ALL_LETTERS)


def _one_hot(index, size):

    vec = [0 for _ in range(size)]
    vec[int(index)] = 1
    return vec


def letter_to_vec(letter):

    index = ALL_LETTERS.find(letter)
    return index


def word_to_indices(word):

    indices = []
    for c in word:
        indices.append(ALL_LETTERS.find(c))
    return indices



