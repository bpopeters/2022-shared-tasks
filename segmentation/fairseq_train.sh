readonly DATA=$1  # example: 2022-shared-tasks/segmentation/eng.word
NAME=$( basename $DATA )  # i.e. eng.word

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
readonly EED=256
readonly EHS=256
readonly DED=256
readonly DHS=256

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
        --dropout="${DROPOUT}" \
        --encoder-embed-dim="${EED}" \
        --encoder-hidden-size="${EHS}" \
        --decoder-embed-dim="${DED}" \
        --decoder-out-embed-dim="${DED}" \
        --decoder-hidden-size="${DHS}" \
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

train "fairseq-checkpoints/baseline"
