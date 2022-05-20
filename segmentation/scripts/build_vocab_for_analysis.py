import sys
import argparse
from itertools import chain
from tokenize import read_tsv, prepare_spm


def main(args:
    vocab = int(vocab)
    data = read_tsv(args.corpus)
    src = data["surface"]
    tgt = data["segment"]

    model_type = "bpe" if args.bpe else "unigram"

    prepare_spm(
        None, args.spm_path, chain(src, tgt), args.vocab, model_type=model_type
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus")
    parser.add_argument("vocab", type=int)
    parser.add_argument("spm_path")
    parser.add_argument("--bpe", action="store_true")
    opt = parser.parse_args()
    main(opt)
