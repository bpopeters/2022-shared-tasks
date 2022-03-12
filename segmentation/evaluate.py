#!/usr/bin/env python

"""
Read dev.tsv and the predictions
"""

import argparse


def read_tsv(path):
    # tsv without header
    split_lines = []
    with open(path) as f:
        for line in f:
            cols = line.strip().split("\t")
            split_lines.append(cols)
    words, segments, codes = zip(*split_lines)
    return words, segments, codes


def read_pred(path):
    with open(path) as f:
        return [line.strip() for line in f]


def error_rate(correct_per_example):
    return 1 - sum(correct_per_example) / len(correct_per_example)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tsv")
    parser.add_argument("pred")
    opt = parser.parse_args()

    words, segments, morph_types = read_tsv(opt.tsv)
    preds = read_pred(opt.pred)
    print(len(words), len(segments), len(morph_types), len(preds))
    assert len(words) == len(segments) == len(morph_types) == len(preds)

    # compute exact matches
    correct = [segs == pred for segs, pred in zip(segments, preds)]
    overall_error_rate = error_rate(correct)
    print("Error Rate (overall):", overall_error_rate)

    # now, stratify by morphological type
    correct_by_type = {mt: [] for mt in set(morph_types)}
    for mt, c in zip(morph_types, correct):
        correct_by_type[mt].append(c)
    for mt, c in correct_by_type.items():
        print("Error Rate ({}):".format(mt), error_rate(c))


if __name__ == "__main__":
    main()
