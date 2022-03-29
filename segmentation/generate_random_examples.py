#!/usr/bin/env python

import argparse
import sys
import random
import numpy as np


def read_dict(path):
    with open(path) as f:
        return dict(line.rstrip().split(" ") for line in f)


def generate_example(vocab, weights, min_len, max_len):
    length = random.randint(min_len, max_len)
    foo = np.random.choice(len(vocab), size=length, p=weights)
    tokens = [vocab[i] for i in foo]
    # tokens = [random.choice(vocab) for i in range(length)]
    return "".join(tokens)


def main(args):
    # read dicts
    # find their intersection
    # generate n 000 examples from them
    # we have counts too, so we can do unigram sampling if we want
    src_dict = read_dict(args.src_dict)
    tgt_dict = read_dict(args.tgt_dict)
    common_keys = set(src_dict) & set(tgt_dict)
    common_dict = {k: src_dict[k] + tgt_dict[k] for k in common_keys}
    vocab = list(common_dict)  # for the simplest thing with a uniform dist

    # np.random.multinomial
    # use it with either uniform or
    if args.unigram:
        class_weights = np.array([common_dict[i] for i in vocab], dtype=float)
        class_weights /= class_weights.sum()
    else:
        class_weights = np.ones(len(vocab)) / len(vocab)

    for _ in range(args.n):
        # generate an example and write it to sys.stdout
        ex = generate_example(vocab, class_weights, args.min_len, args.max_len)
        sys.stdout.write("\t".join([ex, ex]) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-dict")
    parser.add_argument("--tgt-dict")
    parser.add_argument("--n", type=int)
    parser.add_argument("--min-len", type=int, default=3)
    parser.add_argument("--max-len", type=int, default=10)
    parser.add_argument("--unigram", action="store_true")
    opt = parser.parse_args()
    main(opt)
