#!/usr/bin/env

import argparse
import numpy as np
import sentencepiece as spm


def sample_vocab(path, n, min_len):
    # load vocabulary
    pieces = []
    scores = []
    with open(path) as f:
        for line in f:
            piece, score = line.rstrip("\n").split("\t")
            piece = piece.replace("▁", "")
            score = float(score)
            if len(piece) >= min_len and score != 0:
                pieces.append(piece)
                scores.append(score)
    probs = np.exp(np.array(scores))
    probs = probs / probs.sum()
    random_ix = np.random.choice(len(pieces), size=n, p=probs)
    return [pieces[i] for i in random_ix]


def main(args):
    pieces = sample_vocab(args.spm_path + ".vocab", args.n, args.min_len)
    processor = spm.SentencePieceProcessor(model_file=args.spm_path + ".model")
    for piece in pieces:
        src = " ".join(list(piece)) if args.tokenize else piece
        for i in range(args.samples):
            if args.tokenize:
                tgt = " ".join(processor.encode(piece, out_type=str, enable_sampling=True, alpha=0.1))
            else:
                tgt = piece
            
            print("\t".join([src, tgt]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("spm_path", help="not with")
    parser.add_argument("-n", type=int, default=100)
    parser.add_argument("-samples", type=int, default=1)
    parser.add_argument("-min_len", type=int, default=2, help="minimum length for target-side pieces")
    parser.add_argument("-tokenize", action="store_true")
    opt = parser.parse_args()
    main(opt)
