#!/bin/sh

DATA_BIN=$1  # e.g. "new-data-bin/char2piece/ces.word-1000"
NAME=$2  # e.g. "ces-v1000-2022-04-13"
GOLD_PATH=$3
GRID_LOC=$4
shift 4

ENTMAX_ALPHA=1.5
LR=0.001

grid() {
    local -r EMB="$1"; shift
    local -r HID="$1"; shift
    local -r LAYERS="$1" ; shift
    local -r HEADS="$1" ; shift
    for WARMUP in 4000 8000 ; do
        for DROPOUT in 0.1 0.3 ; do
            for BATCH in $@ ; do
                MODEL_DIR="${GRID_LOC}/${NAME}-entmax-minlev-${EMB}-${HID}-${LAYERS}-${HEADS}-${BATCH}-${ENTMAX_ALPHA}-${LR}-${WARMUP}-${DROPOUT}"
                if [ ! -f "${MODEL_DIR}/dev-5.results" ]
                then
                    bash fairseq_train_entmax_transformer.sh $DATA_BIN $NAME $EMB $HID $LAYERS $HEADS $BATCH $ENTMAX_ALPHA $LR $WARMUP $DROPOUT $GRID_LOC
                    bash fairseq_segment.sh $DATA_BIN $MODEL_DIR 1.5 5 $GOLD_PATH
                else
                    echo "skipping ${MODEL_DIR}"
                fi
            done
        done
    done
}

grid 256 1024 3 4 $@
grid 512 2048 3 4 $@
