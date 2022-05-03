readonly DATA_BIN=$1  # example: 2022-shared-tasks/segmentation/eng.word
readonly NAME=$2  # just a special name for the experiment
EMB=$3
HID=$4
LAYERS=$5
BATCH=$6
LR=$7  # note!
GRID_LOC=$8

# Adapted from the SIGMORPHON 2020 script by Kyle Gorman and Shijie Wu.

set -euo pipefail

# Defaults.
readonly SEED=42
readonly CRITERION=cross_entropy
readonly OPTIMIZER=adam
readonly CLIP_NORM=1.
readonly MAX_UPDATE=400000
readonly SAVE_INTERVAL=1
readonly SCHEDULER=reduce_lr_on_plateau
readonly PATIENCE=5
readonly LR_PATIENCE=2
readonly DROPOUT=0.3

MODEL_DIR="${GRID_LOC}/${NAME}-xe-minlev-${EMB}-${HID}-${LAYERS}-${BATCH}-${LR}"

train() {
    local -r CP="$1"; shift
    fairseq-train \
        "${DATA_BIN}" \
        --save-dir="${CP}" \
        --source-lang="src" \
        --target-lang="tgt" \
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
        --optimizer="${OPTIMIZER}" \
        --lr="${LR}" \
        --lr-scheduler="${SCHEDULER}" \
        --lr-patience="${LR_PATIENCE}" \
        --clip-norm="${CLIP_NORM}" \
        --batch-size="${BATCH}" \
        --max-update="${MAX_UPDATE}" \
        --save-interval="${SAVE_INTERVAL}" \
        --patience="${PATIENCE}" \
        --no-epoch-checkpoints \
        --best-checkpoint-metric "lev_dist" \
        --eval-levenshtein \
        --eval-bleu-remove-bpe "sentencepiece" \
        --eval-bleu-args '{"beam_size": 5, "alpha": 1}' \
        "$@"   # In case we need more configuration control.
}

train $MODEL_DIR
