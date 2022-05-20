# DATA=$1  # no "train.tsv" or "dev.tsv" suffix
TRAIN=$1
DEV=$2

CORPUS_WEIGHT=1.0
ANNOT_WEIGHT=0.1

# todo: add hyperparameters

NAME=$( basename $TRAIN .train.tsv)

EXP_NAME="unsup-baselines/morfessor/${NAME}-${CORPUS_WEIGHT}-${ANNOT_WEIGHT}"

mkdir -p $EXP_NAME

# there need to be two files for training: unlabeled corpus and labeled annotations

# copy unsegmented column to a temporary file
CORPUS="${EXP_NAME}/corpus.txt"
cut -f 1 $TRAIN | python scripts/preprocess_morfessor.py > $CORPUS

ANNOT="${EXP_NAME}/annotations.txt"
cut -f 1,2 $TRAIN | sed "s/ @@/‖/g" | sed "s/ /_/g" | sed "s/\t/ /g" | sed "s/‖/ /g" > $ANNOT

BASELINE_PATH="${EXP_NAME}/baseline.gz"
morfessor-train $CORPUS -S $BASELINE_PATH -w $CORPUS_WEIGHT -A $ANNOT -W $ANNOT_WEIGHT

ANALYSIS_PATH="${EXP_NAME}/analysis.tar.gz"
# flatcat-train $BASELINE_PATH -p $PPL_THRESHOLD -w $CORPUS_WEIGHT -A $ANNOT -W $ANNOT_WEIGHT -s $ANALYSIS_PATH

PRED="${EXP_NAME}/baseline.pred"
GUESS="${EXP_NAME}/baseline.guess"
cut -f 1 $DEV | python scripts/preprocess_morfessor.py | morfessor-segment -L $BASELINE_PATH - | python scripts/postprocess_morfessor.py > $PRED
cut -f 1 $DEV | paste - $PRED > $GUESS

python 2022SegmentationST/evaluation/evaluate.py --gold $DEV --guess $GUESS > "${EXP_NAME}/dev.results"
