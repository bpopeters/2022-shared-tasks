#!/bin/sh

DATA=$1
NAME=$( basename $DATA )

VOCAB=$2
OUT_FORMAT=$3  # piece or sample_piece

SAMPLE_TIMES=$4

mkdir -p augmented-data/spm/$OUT_FORMAT
OUT_DATA=augmented-data/spm/$OUT_FORMAT/$NAME-$VOCAB-$SAMPLE_TIMES

SRC_SPM_DATA=$OUT_DATA.train.src.tmp
TGT_SPM_DATA=$OUT_DATA.train.tgt.tmp
cut -f 1 $DATA.train.tsv > $SRC_SPM_DATA
cut -f 2 $DATA.train.tsv > $TGT_SPM_DATA

train() {
    local -r SPM_TRAINING_DATA="$1"; shift # was $SPM_DATA
    local -r PREFIX="$1"; shift  # was $OUT_DATA.spm
    local -r VOCAB_SIZE="$1"; shift
    spm_train --input $SPM_TRAINING_DATA --model_prefix $PREFIX --vocab_size $VOCAB_SIZE --character_coverage 1.0
}

# what differs from train and dev? One always uses piece, the other uses the OUT_FORMAT. Different source and target word lists too.
# maybe it would be better to do all the cutting inside the method so the signature can be simplified
encode() {
    local -r SRC_MODEL="$1"; shift
    local -r TGT_MODEL="$1"; shift
    local -r FORMAT="$1"; shift
    local -r TSV="$1" ; shift

    SRC=$OUT_DATA.src.tmp
    TGT=$OUT_DATA.tgt.tmp
    TAGS=$OUT_DATA.tags.tmp

    cut -f 1 $TSV > $SRC
    cut -f 2 $TSV > $TGT
    cut -f 3 $TSV > $TAGS

    # to be clear, the TGT is what gets segmented
    # hardcoding alpha...not great, probably. I didn't use this for the original swr-8000
    cat $SRC | spm_encode --model $SRC_MODEL --output_format $FORMAT --alpha 0.1 > $SRC.pieces.tmp
    cat $TGT | spm_encode --model $TGT_MODEL --output_format $FORMAT --alpha 0.1 | paste $SRC.pieces.tmp - $TAGS
    # spm_encode --model $OUT_DATA.spm.model --output_format piece < $OUT_DATA.dev.tgt.tmp | paste $OUT_DATA.dev.src.tmp - $OUT_DATA.dev.tags.tmp > $OUT_DATA.dev.tsv
    rm $OUT_DATA.*.tmp
}

train $SRC_SPM_DATA $OUT_DATA.src.spm $VOCAB
train $TGT_SPM_DATA $OUT_DATA.tgt.spm $VOCAB

rm -f $OUT_DATA.train.tsv  # avoid overwriting
for i in $( seq 1 $SAMPLE_TIMES ) ; do
    encode $OUT_DATA.src.spm.model $OUT_DATA.tgt.spm.model $OUT_FORMAT $DATA.train.tsv >> $OUT_DATA.train.tsv
done

encode $OUT_DATA.src.spm.model $OUT_DATA.tgt.spm.model piece $DATA.dev.tsv > $OUT_DATA.dev.tsv


tail -n +4 $OUT_DATA.src.spm.vocab | cut -f 1 | sed "s/$/ 100/g" > $OUT_DATA.src.fairseq.vocab
tail -n +4 $OUT_DATA.tgt.spm.vocab | cut -f 1 | sed "s/$/ 100/g" > $OUT_DATA.tgt.fairseq.vocab


