readonly DATA=$1  # example: 2022-shared-tasks/segmentation/eng.word
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
            cut -f2 "${TSV}" | \
                python tokenize_segments.py > "${TASK}.${NAME}".tgt
        done
    done
}

tsv
