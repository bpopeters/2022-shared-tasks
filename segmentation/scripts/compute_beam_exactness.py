#!/usr/bin/env python

import sys
from collections import defaultdict
from itertools import groupby

# input: sequence of hypothesis lines from output of fairseq-(generate|interactive)


def main():
    # would prefer to do it in streaming fashion, though
    for label, example in groupby(sys.stdin, lambda x: x.split("\t")[0]):
        total_prob = 0.0
        for hyp_line in example:
            _, score, hyp = hyp_line.strip().split("\t")
            total_prob += 2 ** float(score)
        sys.stdout.write("\t".join([label, str(total_prob)]) + "\n")

    '''
    for line in sys.stdin:
        label, score, hyp = line.strip().split("\t")
        prob = 2 ** float(score)
        total_beam_probs[label] += prob
    '''


if __name__ == "__main__":
    main()
