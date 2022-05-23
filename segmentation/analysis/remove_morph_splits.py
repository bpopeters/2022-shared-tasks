#!/usr/bin/env python

import re
import sys

for line in sys.stdin:
    sys.stdout.write(re.sub(r'\s+▁\s*\|\s+', ' ', line))
