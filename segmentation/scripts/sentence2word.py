#!/usr/bin/env python

# take a sentence-level corpus and turn it into a word-level corpus

import argparse
import sys

def get_segmented_words(segment_seq):
    segment_seq = segment_seq.split()
    assert not segment_seq[0].startswith("@@")
    words = []
    for segment in segment_seq:
        if not segment.startswith("@@"):
            new_word = [segment]
            words.append(new_word)
        else:
            words[-1].append(segment)
    return [" ".join(word) for word in words]


def add_context_marker(surface_seq):
    surface_toks = surface_seq.split()
    # for each token in surface_toks
    ret = []
    for i, tok in enumerate(surface_toks):
        labeled_seq = [t if i != j else "<p> {} </p>".format(t)
                       for j, t in enumerate(surface_toks)]
        labeled_string = " ".join(labeled_seq)
        ret.append(labeled_string)
    return ret


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", action="store_true")
    opt = parser.parse_args()
    for line in sys.stdin:
        surface_seq, segment_seq = line.strip().split("\t")
        # these two lines because of problems in the data:
        surface_seq = surface_seq.strip()
        segment_seq = segment_seq.strip()

        if opt.context:
            src_toks = add_context_marker(surface_seq)
        else:
            src_toks = surface_seq.split()
        segmented_words = get_segmented_words(segment_seq)
        if len(src_toks) == len(segmented_words):
            for word, word_segments in zip(src_toks, segmented_words):
                sys.stdout.write("\t".join([word, word_segments]) + "\n")
        else:
            sys.stderr.write(str(src_toks) + "\t" + str(segmented_words) + "\n")
        
            


if __name__ == "__main__":
    main()
