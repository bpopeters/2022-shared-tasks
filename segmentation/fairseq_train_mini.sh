NAME="eng.word"

EMB=512
HID=1024
LAYERS=2
DROPOUT=0.3
BATCH=64
ENTMAX_ALPHA=1.5

# Adapted from the SIGMORPHON 2020 script by Kyle Gorman and Shijie Wu.

set -euo pipefail

# Defaults.
readonly SEED=1917
readonly CRITERION=entmax_loss
readonly OPTIMIZER=adam
readonly LR=1e-3
readonly CLIP_NORM=1.
readonly MAX_UPDATE=50000
readonly SAVE_INTERVAL=1
readonly SCHEDULER=reduce_lr_on_plateau
readonly PATIENCE=5

train() {
    local -r CUR_SIZE="$1" ; shift
    local -r CP="$1"; shift
    fairseq-train \
        "data-bin/mini-data/${NAME}-${CUR_SIZE}" \
        --save-dir="${CP}" \
        --source-lang="${NAME}.src" \
        --target-lang="${NAME}.tgt" \
        --seed="${SEED}" \
        --arch=lstm \
        --encoder-bidirectional \
        --encoder-layers "${LAYERS}" \
        --decoder-layers "${LAYERS}" \
        --dropout="${DROPOUT}" \
        --encoder-embed-dim="${EMB}" \
        --encoder-hidden-size="${HID}" \
        --decoder-embed-dim="${EMB}" \
        --decoder-out-embed-dim="${EMB}" \
        --decoder-hidden-size="${HID}" \
        --share-decoder-input-output-embed \
        --criterion="${CRITERION}" \
        --loss-alpha="${ENTMAX_ALPHA}" \
        --optimizer="${OPTIMIZER}" \
        --lr="${LR}" \
        --lr-scheduler="${SCHEDULER}" \
        --clip-norm="${CLIP_NORM}" \
        --batch-size="${BATCH}" \
        --max-update="${MAX_UPDATE}" \
        --save-interval="${SAVE_INTERVAL}" \
        --patience="${PATIENCE}" \
        --no-epoch-checkpoints \
        "$@"   # In case we need more configuration control.
}

SIZES=(100 500 1000 10000 20000)
for SIZE in "${SIZES[@]}" ; do
    MODEL_DIR="fairseq-checkpoints/mini/${NAME}-${SIZE}-entmax-${EMB}-${HID}-${LAYERS}-${DROPOUT}-${BATCH}-${ENTMAX_ALPHA}"
    train $SIZE $MODEL_DIR
done
