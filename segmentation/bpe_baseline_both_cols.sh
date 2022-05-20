TRAIN=$1
DEV=$2
VOCAB=$3

NAME=$( basename $TRAIN .train.tsv)

EXP_NAME="unsup-baselines/bpe/both-cols/${NAME}-${VOCAB}"

mkdir -p $EXP_NAME

# train spm model
python scripts/build_vocab_for_analysis.py $TRAIN $VOCAB $EXP_NAME/spm.$VOCAB

# segment dev set
cut -f 1 $DEV | spm_encode --model $EXP_NAME/spm.$VOCAB.model --output_format piece | python scripts/spm2out.py > "${EXP_NAME}/spm.out"

# evaluate
cut -f 1 $DEV | paste - "${EXP_NAME}/spm.out" > "${EXP_NAME}/guess"

python 2022SegmentationST/evaluation/evaluate.py --gold $DEV --guess "${EXP_NAME}/guess" > "${EXP_NAME}/dev.results"
