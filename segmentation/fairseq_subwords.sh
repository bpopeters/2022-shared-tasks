readonly DATA_BIN=$1
readonly MODEL_PATH=$2
readonly ENTMAX_ALPHA=$3
readonly BEAM=$4

decode() {
    local -r CP="$1"; shift
    local -r MODE="$1"; shift
    # Fairseq insists on calling the dev-set "valid"; hack around this.
    local -r FAIRSEQ_MODE="${MODE/dev/valid}"
    CHECKPOINT="${CP}/checkpoint_best.pt"
    OUT="${CP}/${MODE}-${BEAM}.subwords.out"
    PRED="${CP}/${MODE}-${BEAM}.subwords.pred"
    # Makes raw predictions.
    fairseq-generate \
        "${DATA_BIN}" \
        --source-lang="src" \
        --target-lang="tgt" \
        --path="${CHECKPOINT}" \
        --gen-subset="${FAIRSEQ_MODE}" \
        --beam="${BEAM}" \
        --alpha="${ENTMAX_ALPHA}" \
	--batch-size 256 \
        > "${OUT}"
    # Extracts the predictions into a TSV file.
    cat "${OUT}" | grep -P '^H-'  | cut -c 3- | sort -n -k 1 | awk -F "\t" '{print $NF}' > $PRED
}

decode $MODEL_PATH dev
