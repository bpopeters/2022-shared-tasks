#!/usr/bin/env python

import argparse
from collections import Counter, defaultdict
from itertools import chain
import sys


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
                if name == "segments":
                    field = field.replace(' @@', '|')
                    field = field.replace(' ', '|')
                data[name].append(field)
    return data


def main(args):
    train = read_tsv(args.train, True)
    train_morphemes = Counter(
        chain.from_iterable([seg.split("|") for seg in train["segments"]])
    )
    # what do I want? a new dataset that covers all the same morphemes as the
    # original one, but that is smaller.
    # map from examples to the morphemes they contain
    # map from morphemes to the examples that contain them
    morph2i = defaultdict(list)
    i2morph = []
    for i, t in enumerate(train["segments"]):
        ex_morphemes = set(t.split("|"))  # does it matter that this is a set?
        i2morph.append(ex_morphemes)
        for m in ex_morphemes:
            morph2i[m].append(i)

    morphemes_covered = Counter()  # avoid duplicates
    examples_added = set()

    thresh = 5

    # how do we order the examples? gaah
    for morph, examples in morph2i.items():
        # list of examples containing this morpheme that are not already in
        # the corpus
        new_ex = [ex for ex in examples if ex not in examples_added]
        if new_ex:
            # we want an example that will a
            best_i = max(new_ex, key=lambda i: sum(max(thresh - morphemes_covered[m], 0) for m in i2morph[i]))
            # for i in new_ex:
            examples_added.add(best_i)
            morphemes_covered.update(i2morph[best_i])
            word = train["word"][best_i]
            segments = train["segments"][best_i]
            category = train["category"][best_i]
            sys.stdout.write("\t".join([word, segments, category]) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train")
    opt = parser.parse_args()
    main(opt)
