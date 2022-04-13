#!/bin/sh

# A more conventional seq2seq baseline for Czech.

DROPOUT=0.3
EMB=512

for f in $@ ; do
    DATA_BIN="new-data-bin/char2char/${f}"
    NAME="2022-04-13"
    for HID in 512 1024 ; do
        for LAYERS in 1 2 ; do
            for BATCH in 32 64 ; do
                for LR in 0.01 0.001 0.0001 ; do
                    bash fairseq_train_xe_improved.sh $DATA_BIN $NAME $EMB $HID $LAYERS $BATCH $LR
                done
            done
        done
    done
done
