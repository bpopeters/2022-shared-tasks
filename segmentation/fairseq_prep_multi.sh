DATA=2022SegmentationST/data

# write (temporary?) src and tgt txt files for each language. Prefixes can end
# in ${LANGUAGE}.word.{src,tgt}.
TXT_DIR=multilingual-data
mkdir -p $TXT_DIR

# names are like "${DATA}/eng.word.${TASK}.tsv"
# it would be nice if this went somewhere other than the project directory
tsv() {
    local -r LANG="$1"; shift
    for TASK in train dev ; do
        TSV="${DATA}/${LANG}.${TASK}.tsv"

        # whitespace-separate the surface column
        # what if we called it "surface" instead of "src"?
        cut -f 1 "${TSV}" | sed "s/ /_/g" | python scripts/split_characters.py > "${TASK}.${LANG}.surface"

        # this uses an @ instead of a pipe. It should use a pipe.
        # but also, it would be nice to just do this with sed
        cut -f 2 "${TSV}" | sed "s/ @@/|/g" | python scripts/split_characters.py > "${TASK}.${LANG}.segment"
    done
}

bin() {
    # todo: fix testpref when it is available
    local -r LANG="$1"; shift
    local -r DEST_DIR="$1"; shift
    local -r DICT="$1"; shift
    fairseq-preprocess \
        --source-lang="${LANG}.surface" \
        --target-lang="${LANG}.segment" \
        --trainpref=train \
        --validpref=dev \
        --testpref=dev \
        --tokenizer=space \
        --thresholdsrc=0 \
        --thresholdtgt=0 \
        --destdir="${DEST_DIR}/${LANG}.surface-${LANG}.segment" \
        --srcdict $DICT \
        --tgtdict $DICT
}

# make directory where binarized data will be stored
DATA_BIN="data-bin/multi-char"
mkdir -p $DATA_BIN

# build a multilingual dictionary from the train sets of all languages
cat $DATA/*.word.train.tsv | cut -f 1,2 | sed "s/ @@/|/g" | sed "s/ /_/g" | python scripts/tsv2dict.py > "${DATA_BIN}/multi.fairseq.vocab"

# for each language, tsv and bin
for NAME in ces.word eng.word fra.word hun.word ita.word lat.word rus.word spa.word ;
do
    tsv $NAME
    bin $NAME $DATA_BIN "${DATA_BIN}/multi.fairseq.vocab"
done
