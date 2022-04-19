#!/usr/bin/env python

import sys
import re
import nltk


def main():
    min_len = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    uniqs = set()
    for line in sys.stdin:
        toks = nltk.word_tokenize(line)
        uniqs.update(toks)
    for word in uniqs:
        if len(word) >= min_len:
            sys.stdout.write(" ".join(list(word)) + "\n")


if __name__ == "__main__":
    main()
