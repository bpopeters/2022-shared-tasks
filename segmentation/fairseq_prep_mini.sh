readonly DATA=$1  # example: 2022-shared-tasks/data/eng.word
NAME=$( basename $DATA )  # i.e. eng.word

SIZES=(100 500 1000 10000 20000)
PATTERNS=(000 001 010 011 100 101 110 111)

train_tsv() {
    local -r CUR_SIZE="$1"; shift
    TASK=train
    TSV="mini-data/${NAME}-${CUR_SIZE}/${NAME}.train.tsv"
    # Separates graphemes with spaces.
    cut -f 1 "${TSV}" | \
        sed 's/./& /g' \
        > "${TASK}.${NAME}".src
    # segments are a little more complicated here.
    # damn I'd rather do this in python
    cut -f2 "${TSV}" | \
        python tokenize_segments.py > "${TASK}.${NAME}".tgt
}

dev_tsv() {
    TASK=dev
    for TSV in "${DATA}.${TASK}.tsv"; do
        # Separates graphemes with spaces.
        cut -f 1 "${TSV}" | \
            sed 's/./& /g' \
            > "${TASK}.${NAME}".src
        # segments are a little more complicated here.
        # damn I'd rather do this in python
        cut -f2 "${TSV}" | \
            python tokenize_segments.py > "${TASK}.${NAME}".tgt
    done
}

bin() {
    local -r CUR_SIZE="$1"; shift
    # todo: fix testpref when it is available
    fairseq-preprocess \
        --source-lang="${NAME}.src" \
        --target-lang="${NAME}.tgt" \
        --trainpref=train \
        --validpref=dev \
        --testpref=dev \
        --tokenizer=space \
        --thresholdsrc=1 \
        --thresholdtgt=1 \
        --srcdict "2022-shared-tasks/segmentation/data-bin/eng.word/dict.eng.word.src.txt"  \
        --tgtdict "2022-shared-tasks/segmentation/data-bin/eng.word/dict.eng.word.tgt.txt" \
        --destdir="data-bin/mini-data/${NAME}-${CUR_SIZE}"
}

mkdir -p mini-data

dev_tsv

for SIZE in "${SIZES[@]}" ; do
    echo $SIZE
    mkdir -p "mini-data/${NAME}-${SIZE}"
    TRAIN_PATH="mini-data/${NAME}-${SIZE}/${NAME}.train.tsv"
    touch $TRAIN_PATH
    for PATTERN in "${PATTERNS[@]}" ; do
        grep $PATTERN "${DATA}.train.tsv" | shuf | head -n $SIZE >> $TRAIN_PATH
    done
    train_tsv $SIZE
    bin $SIZE
    
done


#tsv
#bin
