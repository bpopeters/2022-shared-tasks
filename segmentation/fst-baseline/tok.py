#!/usr/bin/env python

import sys
import re

def add_whitespace(string):
    return " ".join(list(string))

for line in sys.stdin:
    line = re.sub(r' @@', '|', line)
    line = re.sub(r' ', '_', line)
    out_line = add_whitespace(line.strip()) + "\n"
    sys.stdout.write(out_line)
