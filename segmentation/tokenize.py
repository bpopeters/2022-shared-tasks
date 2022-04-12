#!/usr/bin/env python

"""
This is the first part of the shared task pipeline. The input is raw shared
task data stored as tsv files. The output is {train,dev,test}.{src,tgt} files
containing that data, but with tokens separated by whitespace. The supported
tokenization techniques are

1) char: each character is a separate token. Spaces in the raw text are
         replaced by underscores.
2) spm: either learn a new sentencepiece model on the task data, or apply an
        external one.

Regardless of tokenization strategy, this script replaces " @@" with "|" in the
target data.
"""

import argparse
from functools import partial
from itertools import chain

import sentencepiece as spm


def read_tsv(path, category=False):
    # tsv without header
    col_names = ["surface", "segment"]
    if category:
        col_names.append("category")
    data = {name: [] for name in col_names}
    with open(path, encoding='utf-8') as f:
        for line in f:
            fields = line.rstrip("\n").split("\t")
            for name, field in zip(col_names, fields):
                if name == "segment":
                    # replace @@ with | as morpheme boundary character
                    # (without destroying whitespace, because it might be needed
                    # for training an spm model)
                    field = field.replace('@@', '|')
                    # field = field.replace(' ', '|')
                data[name].append(field)
    return data


# ok, that's part of it but not the whole thing
def character_tokenize(string):
    # remove spaces before morpheme boundaries. turn other spaces into
    # underscores.
    string = string.replace(' |', '|')
    string = string.replace(" ", "_")  # maybe only on word level
    return list(string.strip())


def write_tokenized_corpus(path, data, tokenizer):
    with open(path, "w") as f:
        for ex in data:
            toks = tokenizer(ex)
            f.write(" ".join(toks) + "\n")


def build_spm_tokenizer(pretrained_path, new_prefix, train_iter, vocab_size):

    if pretrained_path is not None:
        spm_model_path = pretrained_path
    else:
        spm.SentencePieceTrainer.train(
            sentence_iterator=train_iter,
            model_prefix=new_prefix,
            vocab_size=vocab_size,
            character_coverage=1.0
        )
        spm_model_path = new_prefix + ".model"
    processor = spm.SentencePieceProcessor(model_file=spm_model_path)

    return processor


def main(args):
    data = read_tsv(args.corpus)
    src = data["surface"]
    tgt = data["segment"]

    # cases:
    # char for both
    # spm for one
    # separate spms for each
    # shared spm for both

    if args.src_tok_type == "spm":
        src_processor = build_spm_tokenizer(
            args.pretrained_spm,
            args.new_spm_prefix + ".src",
            chain(src, tgt) if args.shared_data else iter(src),
            args.vocab_size
        )
        # todo: character coverage, alpha hyperparameter
        src_line_tokenizer = partial(
            src_processor.encode,
            out_type=str,
            enable_sampling=args.sample
        )
    else:
        src_line_tokenizer = character_tokenize

    if args.tgt_tok_type == "spm":
        tgt_processor = build_spm_tokenizer(
            args.pretrained_spm,
            args.new_spm_prefix + ".tgt",
            chain(src, tgt) if args.shared_data else iter(tgt),
            args.vocab_size
        )
        tgt_line_tokenizer = partial(
            tgt_processor.encode,
            out_type=str,
            enable_sampling=args.sample
        )
    else:
        tgt_line_tokenizer = character_tokenize

    write_tokenized_corpus(args.out_prefix + ".src", src, src_line_tokenizer)
    write_tokenized_corpus(args.out_prefix + ".tgt", tgt, tgt_line_tokenizer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('corpus', help="tsv file from which to build tokenized data")
    parser.add_argument("--src-tok-type", "-s", default="char", choices=["char", "spm"])
    parser.add_argument("--tgt-tok-type", "-t", default="char", choices=["char", "spm"])
    parser.add_argument('--pretrained-spm', "-p", default=None,
                        help="Path to existing sentencepiece model")
    parser.add_argument('--pretrained-src-spm', default=None,
                        help="Path to existing sentencepiece model")
    parser.add_argument('--pretrained-tgt-spm', default=None,
                        help="Path to existing sentencepiece model")
    parser.add_argument("--new-spm-prefix", "-n", default="m",
                        help="Path to write new spm model to")
    parser.add_argument("--vocab-size", "-v", default=1000, type=int,
                        help="Vocab size if training a new sentencepiece model")
    parser.add_argument("--out-prefix", "-o", default="foo")
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--shared-data", action="store_true")
    opt = parser.parse_args()
    main(opt)
