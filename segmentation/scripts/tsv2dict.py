#!/usr/bin/env python

# STDIN: a TSV with arbitarily many column
# STDOUT: a fairseq-style dictionary file (single space-delimited, no metachars)

import sys
from collections import Counter

vocab = Counter()
for line in sys.stdin:
    for field in line.strip("\t"):
        vocab.update(list(field.strip()))

for char, count in vocab.most_common():
    sys.stdout.write(" ".join([char, str(count)]) + "\n")
