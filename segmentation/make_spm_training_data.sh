#!/bin/sh

DATA=$1
NAME=$( basename $DATA )

VOCAB=$2
OUT_FORMAT=$3  # piece or sample_piece

mkdir -p augmented-data/spm/$OUT_FORMAT
OUT_DATA=augmented-data/spm/$OUT_FORMAT/$NAME-$VOCAB

SPM_DATA=$OUT_DATA.train.tgt.tmp  # also consider training with an external corpus

train() {
    local -r SPM_TRAINING_DATA="$1"; shift # was $SPM_DATA
    local -r PREFIX="$1"; shift  # was $OUT_DATA.spm
    local -r VOCAB_SIZE="$1"; shift
    spm_train --input $SPM_TRAINING_DATA --model_prefix $PREFIX --vocab_size $VOCAB_SIZE --character_coverage 1.0
}

# what differs from train and dev? One always uses piece, the other uses the OUT_FORMAT. Different source and target word lists too.
# maybe it would be better to do all the cutting inside the method so the signature can be simplified
encode() {
    local -r MODEL="$1"; shift
    local -r FORMAT="$1"; shift
    local -r TSV="$1" ; shift

    SRC=$OUT_DATA.src.tmp
    TGT=$OUT_DATA.tgt.tmp
    TAGS=$OUT_DATA.tags.tmp

    cut -f 1 $TSV > $SRC
    cut -f 2 $TSV > $TGT
    cut -f 3 $TSV > $TAGS

    # to be clear, the TGT is what gets segmented
    cat $TGT | spm_encode --model $MODEL --output_format $FORMAT | paste $SRC - $TAGS
    # spm_encode --model $OUT_DATA.spm.model --output_format piece < $OUT_DATA.dev.tgt.tmp | paste $OUT_DATA.dev.src.tmp - $OUT_DATA.dev.tags.tmp > $OUT_DATA.dev.tsv
    rm $OUT_DATA.*.tmp
}

cut -f 1 $DATA.train.tsv > $OUT_DATA.train.src.tmp
cut -f 2 $DATA.train.tsv > $SPM_DATA
cut -f 3 $DATA.train.tsv > $OUT_DATA.train.tags.tmp

cut -f 1 $DATA.dev.tsv > $OUT_DATA.dev.src.tmp
cut -f 2 $DATA.dev.tsv > $OUT_DATA.dev.tgt.tmp
cut -f 3 $DATA.dev.tsv > $OUT_DATA.dev.tags.tmp

train $SPM_DATA $OUT_DATA.spm $VOCAB
encode $OUT_DATA.spm.model $OUT_FORMAT $DATA.train.tsv > $OUT_DATA.train.tsv
encode $OUT_DATA.spm.model piece $DATA.dev.tsv > $OUT_DATA.dev.tsv


tail -n +4 $OUT_DATA.spm.vocab | cut -f 1 | sed "s/$/ 100/g" > $OUT_DATA.fairseq.vocab


