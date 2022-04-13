#!/bin/sh

# A more conventional seq2seq baseline for Czech.

DROPOUT=0.3
EMB=512
ALPHA=1.5

for f in $@ ; do
    DATA_BIN="new-data-bin/char2char/${f}"
    NAME="ces-2022-04-13"
    for HID in 512 1024 ; do
        for LAYERS in 1 2 ; do
            for BATCH in 32 64 ; do
                for LR in 0.001 0.0005 0.0001 ; do
                    bash fairseq_train_improved.sh $DATA_BIN $NAME $EMB $HID $LAYERS $BATCH $ALPHA $LR
                done
            done
        done
    done
done
