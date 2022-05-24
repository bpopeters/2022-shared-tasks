#!/usr/bin/env python

import sys
from collections import defaultdict
from itertools import groupby

# input: sequence of hypothesis lines from output of fairseq-(generate|interactive)
# output: currently a sum for hypotheses.
# I would like to add another column.


def main():
    # would prefer to do it in streaming fashion, though
    for label, example in groupby(sys.stdin, lambda x: x.split("\t")[0]):
        total_prob = 0.0
        best_hyp = 0.0
        for hyp_line in example:
            _, score, hyp = hyp_line.strip().split("\t")
            hyp_prob = 2 ** float(score)
            total_prob += hyp_prob
            best_hyp = max(best_hyp, hyp_prob)
        remaining_mass = 1 - total_prob
        argmax_cert = str(int(best_hyp > remaining_mass))
        sys.stdout.write("\t".join([label, str(total_prob), argmax_cert]) + "\n")

    '''
    for line in sys.stdin:
        label, score, hyp = line.strip().split("\t")
        prob = 2 ** float(score)
        total_beam_probs[label] += prob
    '''


if __name__ == "__main__":
    main()
