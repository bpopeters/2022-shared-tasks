#!/usr/bin/env python

# Script for working with morfessor-segment's output

import sys

def label_suff(line):
    tokens = line.strip().split()
    return " @@".join(tokens)


for line in sys.stdin:
    sys.stdout.write(label_suff(line) + "\n")
