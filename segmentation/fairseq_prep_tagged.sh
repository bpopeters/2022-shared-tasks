# the same as the basic fairseq data, except adding the morphological codes to the beginning of the source sequence

readonly DATA=$1  # example: 2022-shared-tasks/data/eng.word
NAME=$( basename $DATA ).tagged  # i.e. eng.word

tsv() {
    for TASK in train dev ; do
        for TSV in "${DATA}.${TASK}.tsv"; do
            # Separates graphemes with spaces.
            cut -f 1 "${TSV}" | \
                sed 's/./& /g' \
                > "${TASK}.${NAME}".src.tmp
            cut -f 3 "${TSV}" | paste -d " " - "${TASK}.${NAME}".src.tmp > "${TASK}.${NAME}".src
            rm "${TASK}.${NAME}".src.tmp
            # segments are a little more complicated here.
            # damn I'd rather do this in python
            cut -f2 "${TSV}" | \
                python tokenize_segments.py > "${TASK}.${NAME}".tgt
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

tsv
bin
