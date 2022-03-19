readonly DATA=$1  # example: 2022-shared-tasks/data/eng.word
NAME=$( basename $DATA )  # i.e. eng.word
readonly MODEL_PATH=$2
readonly ENTMAX_ALPHA=$3

readonly BEAM=5

decode() {
    local -r CP="$1"; shift
    local -r MODE="$1"; shift
    # Fairseq insists on calling the dev-set "valid"; hack around this.
    local -r FAIRSEQ_MODE="${MODE/dev/valid}"
    CHECKPOINT="${CP}/checkpoint_best.pt"
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
        --gen-subset="${FAIRSEQ_MODE}" \
        --beam="${BEAM}" \
        --alpha="${ENTMAX_ALPHA}" \
        > "${OUT}"
    # Extracts the predictions into a TSV file.
    cat "${OUT}" | grep -P '^H-'  | cut -c 3- | sort -n -k 1 | awk -F "\t" '{print $NF}' | python postprocess_fairseq.py > $PRED
    cut -f 1 "${DATA}.dev.tsv" | paste - $PRED > "${CP}/${MODE}.guess"
    # Applies the evaluation script to the TSV file.
    python evaluate.py 2022SegmentationST/evaluation/ --gold "${DATA}.dev.tsv" --guess "${CP}/${MODE}.guess" --category > "${CP}/${MODE}.results"
}

decode $MODEL_PATH dev
