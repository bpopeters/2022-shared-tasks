#!/usr/bin/env python

import sys
from itertools import groupby

import numpy as np

# input: sequence of hypothesis lines from output of fairseq-(generate|interactive)
# output: currently a sum for hypotheses.
# I would like to add another column.


def entropy(hyp_probs):
    normalized = hyp_probs / hyp_probs.sum()
    log_probs = np.log2(normalized)
    log_probs[np.isinf(log_probs)] = 0
    return -log_probs @ normalized


def main():
    # would prefer to do it in streaming fashion, though
    for label, example in groupby(sys.stdin, lambda x: x.split("\t")[0]):
        hyp_scores = []
        for hyp_line in example:
            _, score, hyp = hyp_line.strip().split("\t")
            hyp_scores.append(float(score))
        hyp_scores = np.array(hyp_scores)
        hyp_probs = 2 ** hyp_scores
        best_prob = hyp_probs.max()
        total_prob = float(hyp_probs.sum())
        remaining_mass = 1 - total_prob
        argmax_cert = str(int(best_prob > remaining_mass))
        beam_entropy = str(float(entropy(hyp_probs)))
        sys.stdout.write("\t".join([label, str(total_prob), beam_entropy, argmax_cert]) + "\n")

    '''
    for line in sys.stdin:
        label, score, hyp = line.strip().split("\t")
        prob = 2 ** float(score)
        total_beam_probs[label] += prob
    '''


if __name__ == "__main__":
    main()
