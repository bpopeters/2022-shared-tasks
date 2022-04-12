readonly DATA_BIN=$1
readonly LANG=$2
readonly MODEL_PATH=$3
readonly ENTMAX_ALPHA=$4

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
        --fixed-dictionary "${DATA_BIN}/multi.fairseq.vocab" \
        --lang-pairs="ces_surface-ces_segment,eng_surface-eng_segment,fra_surface-fra_segment,hun_surface-hun_segment,ita_surface-ita_segment,lat_surface-lat_segment,rus_surface-rus_segment,spa_surface-spa_segment,ces_segment-ces_surface,eng_segment-eng_surface,fra_segment-fra_surface,hun_segment-hun_surface,ita_segment-ita_surface,lat_segment-lat_surface,rus_segment-rus_surface,spa_segment-spa_surface" \
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
