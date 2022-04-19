DATA=$1  # no "train.tsv" or "dev.tsv" suffix

NAME=$( basename $DATA )


DATA=$1  # no "train.tsv" or "dev.tsv" suffix

NAME=$( basename $DATA )

EXP_NAME="unsup-baselines/mor/${NAME}"

mkdir -p $EXP_NAME

# copy unsegmented column to a temporary file
TRAIN="${EXP_NAME}/train.tmp"
cut -f 1 $DATA.train.tsv | python scripts/preprocess_morfessor.py > $TRAIN

MODEL_PATH="${EXP_NAME}/model.bin"
morfessor-train -s $MODEL_PATH $TRAIN

# segment dev set
DEV=$EXP_NAME/dev.tmp
cut -f 1 $DATA.dev.tsv | python scripts/preprocess_morfessor.py > $DEV

# problem with this is that it incorrectly handles entries that have a space in
# them. How do we handle that? One possibility is to replace spaces with "_"
morfessor-segment $DEV -l $MODEL_PATH | python scripts/postprocess_morfessor.py > "${EXP_NAME}/mor.out"

cut -f 1 $DATA.dev.tsv | paste - "${EXP_NAME}/mor.out" > "${EXP_NAME}/guess"

python 2022SegmentationST/evaluation/evaluate.py --gold $DATA.dev.tsv --guess "${EXP_NAME}/guess" > "${EXP_NAME}/dev.results"

rm "${EXP_NAME}/"*".tmp"
