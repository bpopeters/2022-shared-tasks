readonly DATA=$1  # example: 2022-shared-tasks/data/eng.sentence
NAME=$( basename $DATA )  # i.e. eng.sentence

tsv() {
    for TASK in train dev ; do
        for TSV in "${DATA}.${TASK}.tsv"; do
            # replace spaces with underscores, then
            # separate graphemes with spaces.
            cut -f 1 "${TSV}" | \
                python tokenize_segments.py $NAME > "${TASK}.${NAME}".src
            # segments are a little more complicated here.
            # damn I'd rather do this in python
            cut -f2 "${TSV}" | \
                python tokenize_segments.py $NAME > "${TASK}.${NAME}".tgt
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
