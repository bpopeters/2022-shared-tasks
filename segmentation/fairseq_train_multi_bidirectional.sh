# will not use DATA like this
readonly DATA_BIN=$1  # example: 2022-shared-tasks/segmentation/eng.word
EMB=$2
HID=$3
LAYERS=$4
DROPOUT=$5
BATCH=$6
ENTMAX_ALPHA=$7
LR_PATIENCE=$8

# Adapted from the SIGMORPHON 2020 script by Kyle Gorman and Shijie Wu.

set -euo pipefail

# Defaults.
readonly SEED=1917
readonly CRITERION=entmax_loss
readonly OPTIMIZER=adam
readonly LR=1e-3
readonly CLIP_NORM=1.
readonly MAX_UPDATE=200000
readonly SAVE_INTERVAL=1
readonly SCHEDULER=reduce_lr_on_plateau
readonly PATIENCE=10

MODEL_DIR="fairseq-checkpoints/multi/char2char-bidirectional-entmax-${LR_PATIENCE}-${EMB}-${HID}-${LAYERS}-${DROPOUT}-${BATCH}-${ENTMAX_ALPHA}"

train() {
    local -r CP="$1"; shift
    fairseq-train \
        "${DATA_BIN}" \
        --save-dir="${CP}" \
        --task translation_multi_simple_epoch \
        --langs="ces_surface,eng_surface,fra_surface,hun_surface,ita_surface,lat_surface,rus_surface,spa_surface,ces_segment,eng_segment,fra_segment,hun_segment,ita_segment,lat_segment,rus_segment,spa_segment" \
        --lang-pairs="ces_surface-ces_segment,eng_surface-eng_segment,fra_surface-fra_segment,hun_surface-hun_segment,ita_surface-ita_segment,lat_surface-lat_segment,rus_surface-rus_segment,spa_surface-spa_segment,ces_segment-ces_surface,eng_segment-eng_surface,fra_segment-fra_surface,hun_segment-hun_surface,ita_segment-ita_surface,lat_segment-lat_surface,rus_segment-rus_surface,spa_segment-spa_surface" \
        --sampling-method="uniform" \
        --encoder-langtok src \
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
        --lr-patience="${LR_PATIENCE}" \
        --clip-norm="${CLIP_NORM}" \
        --batch-size="${BATCH}" \
        --max-update="${MAX_UPDATE}" \
        --save-interval="${SAVE_INTERVAL}" \
        --patience="${PATIENCE}" \
        --no-epoch-checkpoints \
        "$@"   # In case we need more configuration control.
}

train $MODEL_DIR
