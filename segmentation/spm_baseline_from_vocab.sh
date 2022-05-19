MODEL=$1
DATA=$2 # full path

NAME=$( basename $DATA .dev.tsv)

EXP_NAME="unsup-baselines/spm/piece2piece-vocab/${NAME}"

mkdir -p $EXP_NAME

# segment dev set
cut -f 1 $DATA | spm_encode --model $MODEL --output_format piece | python scripts/spm2out.py > "${EXP_NAME}/spm.out"

# evaluate
cut -f 1 $DATA | paste - "${EXP_NAME}/spm.out" > "${EXP_NAME}/guess"

python 2022SegmentationST/evaluation/evaluate.py --gold $DATA --guess "${EXP_NAME}/guess" > "${EXP_NAME}/dev.results"
