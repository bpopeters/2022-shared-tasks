#!/usr/bin/env python

import sys

for line in sys.stdin:
    sys.stdout.write(" ".join(list(line.strip())) + "\n")
