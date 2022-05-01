DATA=$1  # no "train.tsv" or "dev.tsv" suffix
VOCAB=$2
COL=$3

NAME=$( basename $DATA )

EXP_NAME="unsup-baselines/bpe/${NAME}-${VOCAB}-${COL}"

mkdir -p $EXP_NAME

# copy unsegmented column to a temporary file
TRAIN="${EXP_NAME}/train.tmp"
cut -f $COL $DATA.train.tsv > $TRAIN

# train spm model
spm_train --input $TRAIN --model_prefix $EXP_NAME/spm.$VOCAB --vocab_size $VOCAB --character_coverage 1.0 --model_type "bpe"

# segment dev set
DEV=$EXP_NAME/dev.tmp
cut -f 1 $DATA.dev.tsv > $DEV
spm_encode --model $EXP_NAME/spm.$VOCAB.model --output_format piece < $DEV | python scripts/spm2out.py > "${EXP_NAME}/spm.out"

# evaluate
paste $DEV "${EXP_NAME}/spm.out" > "${EXP_NAME}/guess"

python 2022SegmentationST/evaluation/evaluate.py --gold $DATA.dev.tsv --guess "${EXP_NAME}/guess" > "${EXP_NAME}/dev.results"

rm "${EXP_NAME}/"*".tmp"
