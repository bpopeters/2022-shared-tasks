#!/usr/bin/env python

import sys
from train_000_tagger import Tagger


def read_tsv(path, category):
    # tsv without header
    col_names = ["word", "segments"]
    if category:
        col_names.append("category")
    data = {name: [] for name in col_names}
    with open(path, encoding='utf-8') as f:
        for line in f:
            fields = line.rstrip("\n").split("\t")
            for name, field in zip(col_names, fields):
                data[name].append(field)
    return data


def main():
    tagger = Tagger.load(sys.argv[1])
    guesses = read_tsv(sys.argv[2], False)  # guess file
    unseg_pred = tagger.tag(guesses["word"])
    # now, write to sys.stdout: output is the same as input
    for pred, gold_word, guess in zip(unseg_pred, guesses["word"], guesses["segments"]):
        if pred:
            sys.stdout.write("\t".join([gold_word, gold_word]) + '\n')
        else:
            sys.stdout.write("\t".join([gold_word, guess]) + '\n')


if __name__ == "__main__":
    main()
