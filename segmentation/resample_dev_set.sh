#!/bin/sh

# input is a binary data directory created by fairseq_prep_char2piece.sh
# output is a new directory which contains a version of the dev set with a
# sampled target segmentation. The point of doing this is to evaluate whether
# the loss substantially differs between argmax and sampled segmentations for
# models trained with subword regularization. If they do differ, then we need
# to validate our models a different way!

# takes:
# - path to existing binarized data

EXISTING_DATA_BIN=$1
OUT_PATH=$2
TSV=$3
shift 3

bin() {
    cp "${EXISTING_DATA_BIN}/src.fairseq.vocab" "${OUT_PATH}/src.fairseq.vocab"
    cp "${EXISTING_DATA_BIN}/tgt.fairseq.vocab" "${OUT_PATH}/tgt.fairseq.vocab"
    fairseq-preprocess \
        --source-lang="src" \
        --target-lang="tgt" \
        --validpref="${OUT_PATH}/dev" \
        --tokenizer=space \
        --thresholdsrc=1 \
        --thresholdtgt=1 \
        --srcdict "${EXISTING_DATA_BIN}/src.fairseq.vocab" \
        --tgtdict "${EXISTING_DATA_BIN}/tgt.fairseq.vocab" \
        --destdir="${OUT_PATH}"
}

python tokenize.py "${TSV}" --src-tok-type char --tgt-tok-type spm --existing-tgt-spm "${EXISTING_DATA_BIN}/tgt" --out-dir $OUT_PATH --split dev $@
bin
