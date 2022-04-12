#!/usr/bin/env python

# take a sentence-level corpus and turn it into a word-level corpus

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


def main():
    for line in sys.stdin:
        surface_seq, segment_seq = line.strip().split("\t")
        # these two lines because of problems in the data:
        surface_seq = surface_seq.strip()
        segment_seq = segment_seq.strip()

        surface_toks = surface_seq.split()
        segmented_words = get_segmented_words(segment_seq)
        assert len(surface_toks) == len(segmented_words)
        for word, word_segments in zip(surface_toks, segmented_words):
            sys.stdout.write("\t".join([word, word_segments]) + "\n")
        
            


if __name__ == "__main__":
    main()
