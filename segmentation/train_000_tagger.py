#!/usr/bin/env python

"""
The purpose of this script is to determine whether unsegmentable words can be
identified with simple log-linear classifiers. This is valuable because,
despite their conceptual simplicity, they are the most difficult case for our
models.
"""

import argparse
from itertools import groupby
import pickle

import numpy as np
import torch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


class Tagger(object):

    def __init__(self, clf, vocab):
        self.clf = clf
        self.vocab = vocab

    def tag(self, tokens):
        """
        Return the predicted class of each token
        """
        try:
            line_feats = self.vocab.transform(tokens)
            predictions = self.clf.predict(line_feats)
            return predictions
        except ValueError:
            return []

    def chunk(self, tokens):
        """
        Return a sequence of chunks, where each chunk is a contiguous
        string of tokens with the same tag
        """
        tags = iter(self.tag(tokens))
        # itertools.groupby doesn't sort keys, so it works for
        # finding contiguous chunks
        chunks = groupby(tokens, key=lambda tok: next(tags))
        return [(k, list(g)) for k, g in chunks]

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            return pickle.load(f)


def read_tsv(path, category):
    # tsv without header
    col_names = ["word", "segments"]
    if category:
        col_names.append("category")
    data = {name: [] for name in col_names}
    with open(path, encoding='utf-8') as f:
        for line in f:
            fields = line.rstrip("\n").split("\t")
            for name, field in zip(col_names, fields):
                if name == "segments":
                    field = field.replace(' @@', '|')
                    field = field.replace(' ', '|')
                data[name].append(field)
    return data


def ngram_features(data, n):
    cv = CountVectorizer(
            input='content', analyzer='char',
            lowercase=True, ngram_range=(1, opt.n)
    )
    X = cv.fit_transform(data["word"])
    return X


def batches(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


# so, this seems to work but it's a lot of compute
# Maybe I should write my own classifier module in torch (i.e. solution code)
# and use flair for the features. That might be smarter.
def flair_features(data, batch_size=128):
    import flair
    forward_emb = flair.embeddings.FlairEmbeddings('news-forward')
    backward_emb = flair.embeddings.FlairEmbeddings('news-backward')
    stacked_emb = flair.embeddings.StackedEmbeddings([forward_emb, backward_emb])
    # problem: what if there are words with a space in the middle? Maybe pool.
    sentences = [flair.data.Sentence(word) for word in data["word"]]

    # split the sentence into batches
    for i, batch in enumerate(batches(sentences, batch_size), 1):
        print("Embedding batch {}".format(i))
        stacked_emb.embed(batch)

    X = torch.stack([torch.stack([word.embedding for word in sent]).max(dim=0)[0]
                     for sent in sentences[:50]])
    return X.numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data')
    parser.add_argument('out')
    parser.add_argument('-features', default="ngram", choices=["ngram", "flair"])
    parser.add_argument('-n', type=int, default=3)
    opt = parser.parse_args()

    data = read_tsv(opt.data, True)
    # todo: read tsv file
    # construct examples (column 0) and labels (binarize as label == "000")
    # evaluate on dev: is it any good?
    # build ngram features for words
    if opt.features == "ngram":
        X, embedder = ngram_features(data, opt.n)
    else:
        X = flair_features(data)
    y = np.array([label == "000" for label in data["category"]], dtype=int)
    clf = LogisticRegression(max_iter=1000)
    print(cross_val_score(clf, X, y, cv=5))
    # clf.fit(X, y)
    # clf.sparsify()
    # this last part will break currently with flair
    Tagger(clf, embedder).save(opt.out)


if __name__ == '__main__':
    main()
