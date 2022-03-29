#!/usr/bin/env python

import sys
import re


def postprocess_word(line):
    no_space = "".join(line.strip().split()) + "\n"
    out_line = re.sub(r'@', ' @@', no_space)
    return out_line


def postprocess_sentence(line):
    no_space = "".join(line.strip().split()) + "\n"
    out_line = re.sub(r'_', ' ', no_space)
    out_line = re.sub(r'|', ' @@', out_line)
    return out_line

def main():
    name = sys.argv[1]
    sentence_level = name.endswith("sentence")
    for line in sys.stdin:
        if sentence_level:
            out_line = postprocess_sentence(line)
        else:
            out_line = postprocess_word(line)
        sys.stdout.write(out_line)


if __name__ == "__main__":
    main()
    
