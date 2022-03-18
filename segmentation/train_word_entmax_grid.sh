#!/bin/sh

for f in $@ ; do
    NAME=2022SegmentationST/data/$( basename $f .train.tsv)
    for EMB in 256 512 ; do
        for HID in 512 1024 ; do
            for LAYERS in 1 2 ; do
                for DROPOUT in 0.3 0.5 ; do
                    for ENTMAX_ALPHA in 1.5 2 ; do
                        for BATCH in 256 512 ; do
                            bash fairseq_train_entmax.sh $NAME $EMB $HID $LAYERS $DROPOUT $BATCH $ENTMAX_ALPHA
                        done
                    done
                done
            done
        done
    done
done
