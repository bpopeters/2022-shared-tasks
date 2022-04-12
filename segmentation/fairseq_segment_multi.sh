readonly DATA_BIN=$1
readonly LANG=$2
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
    OUT="${CP}/${MODE}.out"
    PRED="${CP}/${MODE}.pred"
    # Makes raw predictions.
    fairseq-generate \
        $DATA_BIN \
        --source-lang="${LANG}_surface" \
        --target-lang="${LANG}_segment" \
        --path="${CHECKPOINT}" \
        --gen-subset="${FAIRSEQ_MODE}" \
        --beam="${BEAM}" \
        --alpha="${ENTMAX_ALPHA}" \
        --batch-size 256 \
        > "${OUT}"

    # Extracts the predictions into a TSV file.
    # ok, now what do I do about postprocess_fairseq?
    cat "${OUT}" | grep -P '^H-'  | cut -c 3- | sort -n -k 1 | awk -F "\t" '{print $NF}' | sed "s/ //g" | sed "s/_/ /g" | sed "s/|/ @@/g" > $PRED
    cut -f 1 "${DATA}.dev.tsv" | paste - $PRED > "${CP}/${MODE}.guess"
    # Applies the evaluation script to the TSV file.
    python 2022SegmentationST/evaluation/evaluate.py --gold "${DATA}.dev.tsv" --guess "${CP}/${MODE}.guess" > "${CP}/${MODE}.results"
    # python 2022SegmentationST/evaluation/evaluate.py --gold "${DATA}.dev.tsv" --guess "${CP}/${MODE}.guess" --category > "${CP}/${MODE}.tagged.results"
}

decode $MODEL_PATH dev
