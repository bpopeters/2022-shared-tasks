# This script should ONLY be used for word-level experiments because it
# destroys whitespace.
# However, I already ran a huge grid using data using a preprocessing pipeline
# that included this script.
# So, it gets to continue to exist for documentary purposes.

import sys
import re


def tokenize(line):
    # this function regrettably destroys whitespace, which makes its continued
    # use problematic.
    # So, it is deprecated for future use. Remember to change it before you run
    # more word-level experiments!
    line = re.sub(r"@@", "@", line)
    line = re.sub(r" ", "", line)
    return " ".join(list(line))


def tokenize_sentence(line):
    # except the pipe character occurs in the czech data
    line = line.strip()  # to account for (target-side) whitespace problems
    line = re.sub(r" @@", "|", line)
    line = re.sub(r" ", "_", line)
    return " ".join(list(line)) + "\n"


name = sys.argv[1]
sentence_level = name.endswith("sentence")
if not sentence_level:
    sys.stderr.write("WARNING! Deprecated tokenizer!\n")

for line in sys.stdin:
    if sentence_level:
        sys.stdout.write(tokenize_sentence(line))
    else:
        sys.stdout.write(tokenize(line))
