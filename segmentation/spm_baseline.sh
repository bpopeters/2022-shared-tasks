DATA=$1  # no "train.tsv" or "dev.tsv" suffix
VOCAB=$2

NAME=$( basename $DATA )

# copy unsegmented column to a temporary file
TRAIN=$NAME.train.tmp
cut -f 1 $DATA.train.tsv > $TRAIN

# train spm model
spm_train --input $TRAIN --model_prefix spm.$VOCAB --vocab_size $VOCAB --character_coverage 1.0

# segment dev set
DEV=$NAME.dev.tmp
cut -f 1 $DATA.dev.tsv > $DEV
spm_encode --model spm.$VOCAB.model --output_format piece < $DEV | python spm2out.py > $DEV.spm.out

# evaluate
python evaluate.py $DATA.dev.tsv $DEV.spm.out

# copy unsegmented:
# 48790
# 57433

# 4000:
# 51771
# 57433 eng.word.dev.tmp.out

# 8000:
# 49105
# 57433 eng.word.dev.tmp.out

# 16000:
# 47002
# 57433 eng.word.dev.tmp.out

# 32000:
# 45832
# 57433 eng.word.dev.tmp.out

# 64000:
# 45226
# 57433 eng.word.dev.tmp.out

#128000:
# 45341
# 57433 eng.word.dev.tmp.out