#!/usr/bin/env python

import sys
from collections import defaultdict

# input: sequence of hypothesis lines from output of fairseq-(generate|interactive)


def main():
    total_beam_probs = defaultdict(float)
    for line in sys.stdin:
        label, score, hyp = line.strip().split("\t")
        prob = 2 ** float(score)
        total_beam_probs[label] += prob
    print(total_beam_probs)


if __name__ == "__main__":
    main()
