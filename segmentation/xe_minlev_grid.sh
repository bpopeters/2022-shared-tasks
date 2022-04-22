#!/bin/sh

# A more conventional seq2seq baseline for Czech.

DATA_BIN=$1  # e.g. "new-data-bin/char2piece/ces.word-1000"
NAME=$2  # e.g. "ces-v1000-2022-04-13"
GOLD_PATH=$3
GRID_LOC=$4
shift 4

DROPOUT=0.3
EMB=512

for HID in 512 1024 ; do
    for LAYERS in 1 2 ; do
        for BATCH in $@ ; do
            for LR in 0.001 0.0005 0.0001 ; do
                MODEL_DIR="${GRID_LOC}/${NAME}-xe-minlev-${EMB}-${HID}-${LAYERS}-${BATCH}-${LR}"
                if [ ! -f "${MODEL_DIR}/dev-5.results" ]
                then
                    bash fairseq_train_minlev_xe.sh $DATA_BIN $NAME $EMB $HID $LAYERS $BATCH $LR $GRID_LOC
                    bash fairseq_segment.sh $DATA_BIN $MODEL_DIR 1 5 $GOLD_PATH
                else
                    echo "skipping ${MODEL_DIR}"
                fi
            done
        done
    done
done
