#!/usr/bin/env python

import sys
import argparse
import nltk


def read_bitext(path):
    lookup = dict()
    with open(path) as f:
        for line in f:
            raw, segmented = line.strip().split("\t")[:2]
            lookup[raw] = segmented
    return lookup


def main(args):
    segment_dict = read_bitext(args.bitext)
    for line in sys.stdin:
        tokens = nltk.word_tokenize(line.strip())  # I don't want this to be necessary
        # problem: what do you do about punctuation?
        out_tokens = [segment_dict.get(tok, tok) for tok in tokens]
        # unwanted side-effect: punctuation is now separated
        sys.stdout.write(" ".join(out_tokens) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("bitext")
    opt = parser.parse_args()
    main(opt)
