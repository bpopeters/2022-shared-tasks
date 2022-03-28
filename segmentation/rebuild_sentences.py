#!/usr/bin/env python

# you have two things: a stream of line lengths and a stream of 

import sys

if __name__ == "__main__":
    line_lengths = sys.argv[1]
    segmented_words = sys.argv[2]
    with open(line_lengths) as line_f, open(segmented_words) as segment_f:
        for line_length in line_f:
            # for each line in line_f, you will write one line to sys.stdout
            line_length = int(line_length)
            sentence = [segment_f.readline().rstrip("\n") for i in range(line_length)]
            sys.stdout.write(" ".join(sentence) + "\n")
