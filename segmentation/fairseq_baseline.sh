
readonly DATA=$1  # example: 2022-shared-tasks/segmentation/eng.word

# turn a tsv into a pair of bitext files with suffixes .src and .tgt
tsv() {
    for TASK in train dev ; do
        for TSV in "${DATA}.${TASK}.tsv"; do
            # Separates graphemes with spaces.
            cut -f 1 "${TSV}" | \
                sed 's/./& /g' \
                > $(basename ${TSV} .tsv).src
            # segments are a little more complicated here.
            # damn I'd rather do this in python
            cut -f2 "${TSV}" | \
                python tokenize_segments.py \
                > $(basename ${TSV} .tsv).tgt
        done
    done
}

tsv
