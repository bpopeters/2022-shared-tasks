#!/usr/bin/env python

import sys
import re


def main():
    min_count = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    uniqs = set()
    for line in sys.stdin:
        line = re.sub(r'[\.,:;\'\"\(\)]', '', line)
        uniqs.update(tok for tok in line.strip().split())
    for word in uniqs:
        if len(word) >= min_count:
            sys.stdout.write(" ".join(list(word)) + "\n")


if __name__ == "__main__":
    main()
