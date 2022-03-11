import sys
import re


def tokenize(line):
    line = re.sub(r"@@", "@", line)
    line = re.sub(r" ", "", line)
    return " ".join(list(line))


for line in sys.stdin:
    sys.stdout.write(tokenize(line))
