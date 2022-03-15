readonly DATA=$1  # example: 2022-shared-tasks/data/eng.word
NAME=$( basename $DATA ).compounding  # i.e. eng.word

tsv() {
    for TASK in train dev ; do
        for TSV in "${DATA}.${TASK}.tsv"; do
            # Separates graphemes with spaces.
            grep "[01][01]1" "${TSV}" | cut -f 1 | \
                sed 's/./& /g' \
                > "${TASK}.${NAME}".src
            # segments are a little more complicated here.
            # damn I'd rather do this in python
            grep "[01][01]1" "${TSV}" | cut -f2 | \
                python tokenize_segments.py > "${TASK}.${NAME}".tgt
        done
    done
    TSV="${DATA}.dev.tsv"
    TASK=test  # pseudotest
    # Separates graphemes with spaces.
    cat "${TSV}" | cut -f 1 | \
        sed 's/./& /g' \
        > "${TASK}.${NAME}".src
    # segments are a little more complicated here.
    # damn I'd rather do this in python
    cat "${TSV}" | cut -f2 | \
        python tokenize_segments.py > "${TASK}.${NAME}".tgt
}

bin() {
    # todo: fix testpref when it is available
    fairseq-preprocess \
        --srcdict="data-bin/eng.word/dict.eng.word.src.txt" \
        --tgtdict="data-bin/eng.word/dict.eng.word.tgt.txt" \
        --source-lang="${NAME}.src" \
        --target-lang="${NAME}.tgt" \
        --trainpref=train \
        --validpref=dev \
        --testpref=test \
        --tokenizer=space \
        --thresholdsrc=1 \
        --thresholdtgt=1 \
        --destdir="data-bin/${NAME}"
}

tsv
bin
