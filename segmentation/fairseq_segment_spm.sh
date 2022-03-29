readonly DATA=$1  # example: 2022-shared-tasks/data/eng.word
NAME=$( basename $DATA )  # i.e. eng.word
readonly MODEL_PATH=$2
readonly ENTMAX_ALPHA=$3
readonly GOLD_PATH=$4

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
        "data-bin/${NAME}" \
        --source-lang="${NAME}.src" \
        --target-lang="${NAME}.tgt" \
        --path="${CHECKPOINT}" \
        --gen-subset="${FAIRSEQ_MODE}" \
        --beam="${BEAM}" \
        --alpha="${ENTMAX_ALPHA}" \
        --remove-bpe="sentencepiece" \
	--batch-size 256 \
        > "${OUT}"
    # Extracts the predictions into a TSV file.
    cat "${OUT}" | grep -P '^H-'  | cut -c 3- | sort -n -k 1 | awk -F "\t" '{print $NF}' | sed "s/ //g" | sed "s/▁/ /g" | sed "s/^ //g" > $PRED
    cut -f 1 "${DATA}.dev.tsv" | paste - $PRED > "${CP}/${MODE}.guess"
    # Applies the evaluation script to the TSV file.
    python 2022SegmentationST/evaluation/evaluate_word.py --gold $GOLD_PATH --guess "${CP}/${MODE}.guess" > "${CP}/${MODE}.results"
    # python 2022SegmentationST/evaluation/evaluate_word.py --gold "${DATA}.dev.tsv" --guess "${CP}/${MODE}.guess" --category > "${CP}/${MODE}.tagged.results"
}

decode $MODEL_PATH dev
