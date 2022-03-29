#!/bin/sh

DATA=$1
NAME=$( basename $DATA )

VOCAB=$2

mkdir -p augmented-data/spm
OUT_DATA=augmented-data/spm/$NAME-$VOCAB

SPM_DATA=$OUT_DATA.train.tgt.tmp
cut -f 1 $DATA.train.tsv > $OUT_DATA.train.src.tmp
cut -f 2 $DATA.train.tsv > $SPM_DATA
cut -f 3 $DATA.train.tsv > $OUT_DATA.train.tags.tmp

spm_train --input $SPM_DATA --model_prefix $OUT_DATA.spm --vocab_size $VOCAB --character_coverage 1.0

# also consider sample_piece
spm_encode --model $OUT_DATA.spm.model --output_format piece < $SPM_DATA | paste $OUT_DATA.train.src.tmp - $OUT_DATA.train.tags.tmp > $OUT_DATA.train.tsv


cut -f 1 $DATA.dev.tsv > $OUT_DATA.dev.src.tmp
cut -f 2 $DATA.dev.tsv > $OUT_DATA.dev.tgt.tmp
cut -f 3 $DATA.train.tsv > $OUT_DATA.dev.tags.tmp
spm_encode --model $OUT_DATA.spm.model --output_format piece < $OUT_DATA.dev.tgt.tmp | paste $OUT_DATA.dev.src.tmp - $OUT_DATA.dev.tags.tmp > $OUT_DATA.dev.tsv

tail -n +4 $OUT_DATA.spm.vocab | cut -f 1 | sed "s/$/ 100/g" > $OUT_DATA.fairseq.vocab

rm $OUT_DATA.*.tmp
