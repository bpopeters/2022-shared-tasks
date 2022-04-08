readonly DATA=$1  # example: 2022-shared-tasks/data/eng.word
NAME=$( basename $DATA )  # i.e. eng.word

tsv() {
    for TASK in train dev ; do
        for TSV in "${DATA}.${TASK}.tsv"; do
            # Separates graphemes with spaces.
            cut -f 1 "${TSV}" | \
                sed 's/./& /g' \
                > "${TASK}.${NAME}".src
            # segments are a little more complicated here.
            # damn I'd rather do this in python
            cut -f2 "${TSV}" > "${TASK}.${NAME}".tgt
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
        --srcdict "${DATA}.src.fairseq.vocab"  \
        --tgtdict "${DATA}.tgt.fairseq.vocab" \
        --destdir="data-bin/${NAME}"
}

tsv
bin
