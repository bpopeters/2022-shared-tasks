
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
readonly MAX_UPDATE=4000
readonly SAVE_INTERVAL=5
readonly EED=256
readonly EHS=256
readonly DED=256
readonly DHS=256

# Hyperparameters to be tuned.
readonly BATCH=256
readonly DROPOUT=.3

# Prediction options.
readonly BEAM=5

# turn a tsv into a pair of bitext files with suffixes .src and .tgt
tsv() {
    for TASK in train dev ; do
        for TSV in "${DATA}.${TASK}.tsv"; do
            # Separates graphemes with spaces.
            cut -f 1 "${TSV}" | \
                sed 's/./& /g' \
                > "${TASK}.${NAME}".src
            # segments are a little more complicated here.
            # damn I'd rather do this in python
            cut -f2 "${TSV}" | \
                python scripts/tokenize_segments.py > "${TASK}.${NAME}".tgt
        done
    done
}

bin() {
    # todo: fix testpref when it is available
    fairseq-preprocess \
        --source-lang="${NAME}.src" \
        --target-lang="${NAME}.tgt" \
        --trainpref=train \
        --validpref=dev \
        --testpref=dev \
        --tokenizer=space \
        --thresholdsrc=1 \
        --thresholdtgt=1 \
        --destdir="data-bin/${NAME}"
}

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

decode() {
    local -r CP="$1"; shift
    local -r MODE="$1"; shift
    # Fairseq insists on calling the dev-set "valid"; hack around this.
    local -r FAIRSEQ_MODE="${MODE/dev/valid}"
    for CHECKPOINT in $(ls ${CP}/checkpoint[1-9]*.pt 2> /dev/null); do
        RES="${CHECKPOINT/.pt/-${MODE}.res}"
        # Don't overwrite an existing prediction file.
        if [[ -f "${RES}" ]]; then
            continue
        fi
        echo "Evaluating into ${RES}"
        OUT="${CP}/${MODE}.out"
        PRED="${CP}/${MODE}.pred"
        # Makes raw predictions.
        fairseq-generate \
            "data-bin/${NAME}" \
            --source-lang="${NAME}.src" \
            --target-lang="${NAME}.tgt" \
            --path="${CHECKPOINT}" \
            --seed="${SEED}" \
            --gen-subset="${FAIRSEQ_MODE}" \
            --beam="${BEAM}" \
            --no-progress-bar \
            > "${OUT}"
        # Extracts the predictions into a TSV file.
        cat "${OUT}" | grep '^H-' | cut -f3 > $PRED
        # Applies the evaluation script to the TSV file.
        python evaluate.py "${DATA}.dev.tsv" "${PRED}"
    done
}

main() {
    bash fairseq_prep.sh $DATA
    bash fairseq_train.sh $DATA
    bash fairseq_segment.sh $DATA
}

main
