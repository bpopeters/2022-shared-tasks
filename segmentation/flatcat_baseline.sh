# DATA=$1  # no "train.tsv" or "dev.tsv" suffix
TRAIN=$1
DEV=$2

CORPUS_WEIGHT=1.0
ANNOT_WEIGHT=0.1
PPL_THRESHOLD=100

# todo: add hyperparameters

NAME=$( basename $TRAIN .train.tsv)

EXP_NAME="unsup-baselines/flatcat/${NAME}-${CORPUS_WEIGHT}-${ANNOT_WEIGHT}-${PPL_THRESHOLD}"

mkdir -p $EXP_NAME

# there need to be two files for training: unlabeled corpus and labeled annotations

# copy unsegmented column to a temporary file
CORPUS="${EXP_NAME}/corpus.txt"
cut -f 1 $TRAIN | python scripts/preprocess_morfessor.py > $CORPUS

ANNOT="${EXP_NAME}/annotations.txt"
cut -f 1,2 $TRAIN | sed "s/ @@/‖/g" | sed "s/ /_/g" | sed "s/\t/ /g" | sed "s/‖/ /g" > $ANNOT

BASELINE_PATH="${EXP_NAME}/baseline.gz"
morfessor-train $CORPUS -S $BASELINE_PATH -w $CORPUS_WEIGHT -A $ANNOT -W $ANNOT_WEIGHT

ANALYSIS_PATH="${EXP_NAME}/analysis.targ.gz"
flatcat-train $BASELINE_PATH -p $PPL_THRESHOLD -w $CORPUS_WEIGHT -A $ANNOT -W $ANNOT_WEIGHT -s $ANALYSIS_PATH

cut -f 1 $DEV | flatcat-segment $ANALYSIS_PATH - -o "${EXP_NAME}/flatcat.pred" --remove-nonmorphemes


# MODEL_PATH="${EXP_NAME}/model.bin"
# morfessor-train -s $MODEL_PATH $TRAIN

# segment dev set
#DEV=$EXP_NAME/dev.tmp
#cut -f 1 $DATA.dev.tsv | python scripts/preprocess_morfessor.py > $DEV

# problem with this is that it incorrectly handles entries that have a space in
# them. How do we handle that? One possibility is to replace spaces with "_"
#morfessor-segment $DEV -l $MODEL_PATH | python scripts/postprocess_morfessor.py > "${EXP_NAME}/mor.out"

#cut -f 1 $DATA.dev.tsv | paste - "${EXP_NAME}/mor.out" > "${EXP_NAME}/guess"

#python 2022SegmentationST/evaluation/evaluate.py --gold $DATA.dev.tsv --guess "${EXP_NAME}/guess" > "${EXP_NAME}/dev.results"
