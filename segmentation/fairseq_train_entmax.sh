readonly DATA=$1  # example: 2022-shared-tasks/segmentation/eng.word
NAME=$( basename $DATA )  # i.e. eng.word
EMB=$2
HID=$3
LAYERS=$4
DROPOUT=$5
BATCH=$6
ENTMAX_ALPHA=$7

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

MODEL_DIR="fairseq-checkpoints/${NAME}-entmax-${EMB}-${HID}-${LAYERS}-${DROPOUT}-${BATCH}-${ENTMAX_ALPHA}"

train() {
    local -r CP="$1"; shift
    fairseq-train \
        "data-bin/${NAME}" \
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

train $MODEL_DIR
