#!/usr/bin/env python

import sys

for line in sys.stdin:
    sys.stdout.write(str(len(line.strip().split())) + "\n")
