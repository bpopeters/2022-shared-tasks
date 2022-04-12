#!/usr/bin/env python

import sys
import re


if __name__ == "__main__":
    for line in sys.stdin:
        sys.stdout.write(re.sub(r' ', '_', line))
