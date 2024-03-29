readonly DATA=$1  # example: 2022-shared-tasks/segmentation/eng.word
NAME=$( basename $DATA )  # i.e. eng.word
EMB=$2
HID=$3
LAYERS=$4
DROPOUT=$5

# Adapted from the SIGMORPHON 2020 script by Kyle Gorman and Shijie Wu.

set -euo pipefail

# Defaults.
readonly SEED=1917
readonly CRITERION=label_smoothed_cross_entropy
readonly LABEL_SMOOTHING=.1
readonly OPTIMIZER=adam
readonly LR=1e-3
readonly CLIP_NORM=1.
readonly MAX_UPDATE=20000
readonly SAVE_INTERVAL=5

# Hyperparameters to be tuned.
readonly BATCH=256
readonly DROPOUT=.3

train() {
    local -r CP="$1"; shift
    fairseq-train \
        "data-bin/${NAME}" \
        --save-dir="${CP}" \
        --source-lang="${NAME}.src" \
        --target-lang="${NAME}.tgt" \
        --disable-validation \
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
        --label-smoothing="${LABEL_SMOOTHING}" \
        --optimizer="${OPTIMIZER}" \
        --lr="${LR}" \
        --clip-norm="${CLIP_NORM}" \
        --batch-size="${BATCH}" \
        --max-update="${MAX_UPDATE}" \
        --save-interval="${SAVE_INTERVAL}" \
        "$@"   # In case we need more configuration control.
}

train "fairseq-checkpoints/${NAME}-xent-${EMB}-${HID}-${LAYERS}-${DROPOUT}"
