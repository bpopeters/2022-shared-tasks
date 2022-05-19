import sys
from itertools import chain
from tokenize import read_tsv, prepare_spm


def main(corpus, vocab, spm_path, *kwargs):
    vocab = int(vocab)
    data = read_tsv(corpus)
    src = data["surface"]
    tgt = data["segment"]

    prepare_spm(None, spm_path, chain(src, tgt), vocab)


if __name__ == "__main__":
    main(*sys.argv[1:])
