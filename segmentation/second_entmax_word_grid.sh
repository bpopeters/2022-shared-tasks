#!/bin/sh

# or; A Fact-Finding Mission

DROPOUT=0.3
EMB=512
ENTMAX_ALPHA=1.5  # can repeat later with sparsemax if results are interesting.

for f in $@ ; do
    NAME=2022SegmentationST/data/$( basename $f .train.tsv)
    for HID in 512 1024 ; do
        for LAYERS in 1 2 ; do
            for BATCH in 32 64 ; do
                bash fairseq_train_entmax.sh $NAME $EMB $HID $LAYERS $DROPOUT $BATCH $ENTMAX_ALPHA
                bash fairseq_train_fyls.sh $NAME $EMB $HID $LAYERS $DROPOUT $BATCH $ENTMAX_ALPHA 0.01
            done
        done
    done
done
