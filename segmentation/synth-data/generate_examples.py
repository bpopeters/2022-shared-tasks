"""
A script for generating hallucinated training data for the SIGMORPHON 2020
shared task.

The code is adapted from Anastasopoulos and Neubig's submission
to the SIGMORPHON 2019 shared task:
- code: https://github.com/antonisa/inflection
- paper: https://www.aclweb.org/anthology/D19-1091.pdf
"""

import sys
import argparse
from itertools import chain, groupby
from collections import Counter

import numpy as np

import align


def read_data(filename):
    inputs = []
    outputs = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line:
                fields = line.split("\t")
                src, trg = fields[:2]
                inputs.append(src.strip())
                outputs.append(trg.strip().replace("@@", "|"))

    return inputs, outputs


def joint_segments(src, tgt):
    """
    src, tgt: equal-length lists
    returns a list of tuples, each corresponding to a (src, tgt) segment
    """
    # this might not be quite correct: see scummered
    start_ix = [0] + [i for i, char in enumerate(tgt) if char == "|"]
    end_ix = [i - 1 for i in start_ix[1:]] + [len(tgt)]
    segments = []
    for start, end in zip(start_ix, end_ix):
        # ok, nice.
        src_segment = "".join(src[start: end]).strip()
        tgt_segment = "".join(tgt[start: end]).strip().replace("|", "")
        segments.append((src_segment, tgt_segment))
    return tuple(segments)


def learn_alignments(src, tgt):
    temp = [(s, t) for s, t in zip(src, tgt)]
    # aligned returns pairs of strings with spaces for null alignments
    aligned = align.Aligner(temp).alignedpairs
    return [joint_segments(s, t) for (s, t) in aligned]


def make_skeleton(segments):
    """
    segments is a sequence of tuples, where each is a (surface, segmented) pair
    This function returns a genericized version of the segment sequence, with
    the longest (measuring by surface length) segment replaced by a generic
    symbol.
    """
    i, longest = max(enumerate(segments), key=lambda x: len(x[1][0]))
    # future: more sophisticated rules
    skeleton = tuple(segment if segment != longest else ("_", "_")
                     for segment in segments)
    return skeleton


def make_vocab(src, tgt):
    # make the intersection
    src_vocab = Counter(chain(*src))
    tgt_vocab = Counter(chain(*tgt))
    common_keys = set(src_vocab) & set(tgt_vocab)
    common_keys.remove(" ")
    merged = Counter({k: src_vocab[k] + tgt_vocab[k] for k in common_keys})
    vocab, vocab_counts = zip(*merged.most_common())
    vocab_counts = np.array(vocab_counts)
    vocab_p = vocab_counts / vocab_counts.sum()

    return vocab, vocab_p


def generate_batch(batch_size, vocab, vocab_p, skeletons, skeleton_p):
    # sample a length (everything in batch is the same length)
    min_length = 3
    max_length = 10
    lengths = np.random.randint(min_length, max_length, size=batch_size)
    # sample a bunch of tokens in the right shape
    random_tokens = np.random.choice(
        len(vocab_p),
        size=(batch_size, max_length),
        p=vocab_p
    )

    # sample a bunch of skeletons
    batch_skel = np.random.choice(
        len(skeletons),
        size=batch_size,
        p=skeleton_p
    )

    fake_examples = []
    for skel_i, toks, length in zip(batch_skel, random_tokens, lengths):
        morpheme = "".join([vocab[char] for char in toks][:length])
        src_segs, tgt_segs = zip(*skeletons[skel_i])
        src = "".join(src_segs).replace("_", morpheme)
        tgt = " @@".join(tgt_segs).replace("_", morpheme)
        fake_examples.append((src, tgt))
    return fake_examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("datapath", type=str, help="path to data")
    parser.add_argument("--examples", default=10000, type=int,
                        help="number of examples to hallucinate (def: 10000)")
    parser.add_argument("--batch_size", default=1000, type=int)
    args = parser.parse_args()

    data_path = args.datapath

    N = args.examples

    # read in a tsv of examples (todo: handle categories)
    true_src, true_trg = read_data(data_path)
    vocab, vocab_p = make_vocab(true_src, true_trg)

    # learn alignments between src and tgt segments
    aligned_segments = learn_alignments(true_src, true_trg)

    # turn alignments between specific strings into
    # skeletons that new examples can be generated from
    # It would be nice to stratify this by category as well.
    unique_skeletons = Counter(make_skeleton(ex) for ex in aligned_segments)
    skeletons, skeleton_counts = zip(*unique_skeletons.most_common())
    skeleton_counts = np.array(skeleton_counts)
    skeleton_p = skeleton_counts / skeleton_counts.sum()

    for i in range(N // args.batch_size):
        batch = generate_batch(
            args.batch_size, vocab, vocab_p, skeletons, skeleton_p
        )
        for ex in batch:
            sys.stdout.write("\t".join(ex) + "\n")


if __name__ == "__main__":
    main()
