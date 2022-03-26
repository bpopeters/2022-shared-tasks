# This script should ONLY be used for word-level experiments because it
# destroys whitespace.
# However, I already ran a huge grid using data using a preprocessing pipeline
# that included this script.
# So, it gets to continue to exist for documentary purposes.

import sys
import re


def tokenize(line):
    line = re.sub(r"@@", "@", line)
    line = re.sub(r" ", "", line)
    return " ".join(list(line))


for line in sys.stdin:
    sys.stdout.write(tokenize(line))
