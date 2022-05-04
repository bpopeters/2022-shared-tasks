readonly DATA_BIN=$1  # example: 2022-shared-tasks/segmentation/eng.word
readonly NAME=$2  # just a special name for the experiment
EMB=$3
HID=$4
LAYERS=$5
BATCH=$6
ENTMAX_ALPHA=$7
LR=$8  # note!
WARMUP=$9
GRID_LOC="${10}"

# warmup steps?

set -euo pipefail

# Defaults.
readonly SEED=42
readonly CRITERION=entmax_loss
readonly OPTIMIZER=adam
readonly CLIP_NORM=1.
readonly MAX_UPDATE=400000
readonly SAVE_INTERVAL=1
readonly LR_SCHEDULER=inverse_sqrt
readonly WARMUP_INIT_LR=1e-7
readonly PATIENCE=5
readonly DROPOUT=0.3
readonly ENCODER_ATTENTION_HEADS=8
readonly DECODER_ATTENTION_HEADS=8

MODEL_DIR="${GRID_LOC}/${NAME}-entmax-minlev-${EMB}-${HID}-${LAYERS}-${BATCH}-${ENTMAX_ALPHA}-${LR}-${WARMUP}"

train() {
    local -r CP="$1"; shift
    local -r ENCODER_LAYERS="$1"; shift
    local -r DECODER_LAYERS="$1"; shift
    fairseq-train \
        "${DATA_BIN}" \
        --save-dir="${CP}" \
        --source-lang="src" \
        --target-lang="tgt" \
        --seed="${SEED}" \
        --arch=transformer \
        --attention-dropout="${DROPOUT}" \
        --activation-dropout="${DROPOUT}" \
        --activation-fn="${ACTIVATION_FN}" \
        --encoder-embed-dim="${EMB}" \
        --encoder-ffn-embed-dim="${HID}" \
        --encoder-layers="${ENCODER_LAYERS}" \
        --encoder-attention-heads="${ENCODER_ATTENTION_HEADS}" \
        --encoder-normalize-before \
        --decoder-embed-dim="${EMB}" \
        --decoder-ffn-embed-dim="${HID}" \
        --decoder-layers="${DECODER_LAYERS}" \
        --decoder-attention-heads="${DECODER_ATTENTION_HEADS}" \
        --decoder-normalize-before \
        --share-decoder-input-output-embed \
        --criterion="${CRITERION}" \
        --loss-alpha="${ENTMAX_ALPHA}" \
        --optimizer="${OPTIMIZER}" \
        --lr="${LR}" \
        --lr-scheduler="${LR_SCHEDULER}" \
        --warmup-init-lr="${WARMUP_INIT_LR}" \
        --warmup-updates="${WARMUP}" \
        --clip-norm="${CLIP_NORM}" \
        --max-tokens="${BATCH}" \
        --max-update="${MAX_UPDATE}" \
        --save-interval="${SAVE_INTERVAL}" \
        --patience="${PATIENCE}" \
        --no-epoch-checkpoints \
        --best-checkpoint-metric "lev_dist" \
        --eval-levenshtein \
        --eval-bleu-remove-bpe "sentencepiece" \
        --eval-bleu-args '{"beam_size": 5, "alpha": 1.5}' \
        "$@"   # In case we need more configuration control.
}

train $MODEL_DIR $LAYERS $LAYERS
