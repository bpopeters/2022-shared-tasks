DATA=$1  # no "train.tsv" or "dev.tsv" suffix
VOCAB=$2

NAME=$( basename $DATA )

# copy unsegmented column to a temporary file
TRAIN=$NAME.train.tmp
cut -f 1 $DATA.train.tsv > $TRAIN

# train spm model
spm_train --input $TRAIN --model_prefix bpe.$VOCAB --vocab_size $VOCAB --character_coverage 1.0 --model_type bpe

# segment dev set
DEV=$NAME.dev.tmp
cut -f 1 $DATA.dev.tsv > $DEV
spm_encode --model bpe.$VOCAB.model --output_format piece < $DEV | python spm2out.py > $DEV.out

# evaluate
ERRORS=$( cut -f 2 $DATA.dev.tsv | diff -y - eng.word.dev.tmp.out | grep "|" | wc -l )
TOTAL=$( wc -l eng.word.dev.tmp.out)
echo $ERRORS
echo $TOTAL

# copy unsegmented:
# 48790
# 57433

# 4000:
# 56662
# 57433 eng.word.dev.tmp.out

# 8000:
# 55875
# 57433 eng.word.dev.tmp.out

# 16000:
# 54996
# 57433 eng.word.dev.tmp.out

# 32000:
# 54089
# 57433 eng.word.dev.tmp.out

# 64000:
# 53476
# 57433 eng.word.dev.tmp.out

#128000:
# 53212
# 57433 eng.word.dev.tmp.out
