#!/bin/sh

DATA_PATH=$1
MODEL_PATH=$2
TEST_PATH=$3
shift 3

# todo: add case for character-level segmentation
preptok() {
    cat $1 | spm_encode --model "${DATA_PATH}/src.model"
}

# tokenize test examples: this is either whitespace segmentation or applying
# a sentencepiece model to them

preptok $TEST_PATH | \
    fairseq-interactive \
        $DATA_PATH \
        --path $MODEL_PATH \
        --source-lang src \
        --target-lang tgt \
        --beam 5 \
        --alpha 1.5  \
        --batch-size 256 \
        --buffer-size 256 | \
    grep -P '^H-'  | \
    cut -c 3- | \
    awk -F "\t" '{print $NF}' | \
    python detokenize.py | \
    paste $TEST_PATH -
    
