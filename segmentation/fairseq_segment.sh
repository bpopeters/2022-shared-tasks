readonly DATA=$1  # example: 2022-shared-tasks/segmentation/eng.word
NAME=$( basename $DATA )  # i.e. eng.word

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

decode "fairseq-checkpoints/baseline" dev
