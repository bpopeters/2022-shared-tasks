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

import sys
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
                    field = field.replace(' @@', '|')
                    # field = field.replace(' ', '|')
                data[name].append(field)
    return data


# ok, that's part of it but not the whole thing
def character_tokenize(string):
    string = string.replace(" ", "_")  # maybe only on word level
    return list(string.strip())


def write_tokenized_corpus(path, data, tokenizer):
    with open(path, "w") as f:
        for ex in data:
            toks = tokenizer(ex)
            f.write(" ".join(toks) + "\n")


def main(args):
    # this doesn't actually make much sense if you think about it because
    # args.corpus is a tsv

    # regardless of tokenization strategy, "|" is the morpheme separator.
    # should this take paths for all of the corpora? I think not. You can run
    # the script separately for train/dev/test: if using spm, you can create
    # the model by running it for the training data first and then 
    data = read_tsv(args.corpus)
    src = data["surface"]
    tgt = data["segment"]

    # ok, we have the src and tgt.
    if args.tok_type == "spm":
        # either train a new spm model or load an existing one
        if args.spm_model is not None:
            spm_model_path = args.spm_model
        else:
            # things get interesting here: do we want to train the spm model
            # one one field or both?
            spm.SentencePieceTrainer.train(
                sentence_iterator=chain(src, tgt),
                model_prefix='m',
                vocab_size=args.vocab_size
            )
            spm_model_path = "m.model"
        processor = spm.SentencePieceProcessor(model_file=spm_model_path)

        # other line_tokenizer arguments apply as well
        # should I
        line_tokenizer = partial(processor.encode, out_type=str, enable_sampling=args.sample)
    else:
        line_tokenizer = character_tokenize

    # todo: do not hardcode these names
    write_tokenized_corpus(args.out_prefix + ".src", src, line_tokenizer)
    write_tokenized_corpus(args.out_prefix + "foo.tgt", tgt, line_tokenizer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('corpus', help="tsv file from which to build tokenized data")
    parser.add_argument("--tok-type", "-t", default="char", choices=["char", "spm"])
    parser.add_argument("--spm-model", "-s", default=None,
                        help="Path to existing sentencepiece model")
    parser.add_argument("--vocab-size", "-v", default=1000, type=int,
                        help="Vocab size if training a new sentencepiece model")
    parser.add_argument("--out-prefix", "-o", default="foo")
    parser.add_argument("--sample", action="store_true")
    opt = parser.parse_args()
    main(opt)
