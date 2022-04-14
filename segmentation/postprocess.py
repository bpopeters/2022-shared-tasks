#!/usr/bin/env python

import sys
import argparse


def postprocess_chars(line):
    line = "".join(line.strip().split(" "))
    line = line.replace("_", " ")
    line = line.replace("|", " @@")
    return line


def main(args):
    for line in sys.stdin:
        sys.stdout.write(postprocess_chars(line) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--spm-model", default=None)
    opt = parser.parse_args()
    main(opt)
