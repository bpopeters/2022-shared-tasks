#!/usr/bin/env python

import sys
import re


if __name__ == "__main__":
    for line in sys.stdin:
        line = re.sub(r'_', ' ', line)
        sys.stdout.write(re.sub(r' ', ' @@', line))
