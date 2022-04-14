#!/bin/sh

# A more conventional seq2seq baseline for Czech.

DATA_BIN=$1  # e.g. "new-data-bin/char2piece/ces.word-1000"
NAME=$2  # e.g. "ces-v1000-2022-04-13"
shift 2

DROPOUT=0.3
EMB=512
ALPHA=1.5

for HID in 512 1024 ; do
    for LAYERS in 1 2 ; do
        for BATCH in $@ ; do
            for LR in 0.001 0.0005 0.0001 ; do
                bash fairseq_train_improved.sh $DATA_BIN $NAME $EMB $HID $LAYERS $BATCH $ALPHA $LR
            done
        done
    done
done
