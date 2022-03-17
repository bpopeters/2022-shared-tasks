#!/usr/bin/env python

import sys
import re


def main():
    uniqs = set()
    for line in sys.stdin:
        line = re.sub(r'[\.,:;\'\"\(\)]', '', line)
        uniqs.update(tok for tok in line.strip().split())
    for word in uniqs:
        if len(word) > 3:
            sys.stdout.write(word + "\n")


if __name__ == "__main__":
    main()
