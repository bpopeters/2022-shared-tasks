#!/usr/bin/env python

from collections import Counter
import sys
import numpy as np


def main(args):
    # count tokens
    counts = Counter()
    for line in sys.stdin:
        tokens = line.strip().split()
        counts.update(tokens)

    # K classes (two views of what K is, I guess: either an external vocab or the
    # actually occurring types.

    # (1/2) sum_i=1..K[ abs(p_i - 1\K) ]
    K = int(args[1]) if len(args) > 1 else len(counts)
    counts_array = np.array(list(counts.values()))
    n_tokens = counts_array.sum()
    p = counts_array / n_tokens

    D = np.abs(p - 1 / K).sum() / 2

    print(D)


if __name__ == "__main__":
    main(sys.argv)
