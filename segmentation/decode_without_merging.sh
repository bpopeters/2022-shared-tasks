#!/bin/sh

DATA_PATH=$1
MODEL_PATH=$2
TEST_PATH=$3
shift 3

# todo: add case for character-level segmentation
preptok() {
    if [ ! -f "${DATA_PATH}/src.model" ]
    then
        cat $1 | python scripts/tokenize_src_chars.py
    else
        cat $1 | spm_encode --model "${DATA_PATH}/src.model"
    fi
}

# tokenize test examples: this is either whitespace segmentation or applying
# a sentencepiece model to them

preptok $TEST_PATH | \
    fairseq-interactive \
        $DATA_PATH \
        --path $MODEL_PATH \
        --source-lang src \
        --target-lang tgt \
        --batch-size 256 \
        --buffer-size 256 \
        "$@" | \
    grep -P '^H-'  | \
    cut -c 3- | \
    awk -F "\t" '{print $NF}' | \
    paste $TEST_PATH -
