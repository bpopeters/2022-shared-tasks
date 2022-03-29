

WORD_MODEL=$1
DATA=$2
NAME=$( basename $DATA )
LANG=$( basename $DATA .sentence)

# General idea:
# - get unique types from corpus (applying some kind of length filtering)
# fix these paths
# python unique_types.py < other-data/europarl/Europarl.en-hu.en > other-data/train.eng.words.src

cut -f 1 "${DATA}.dev.tsv" | python unique_types.py  > $NAME.dev.src

# - segment these types (this will require preprocessing them first)
fairseq-interactive \
    "data-bin/${LANG}.word" \
    --path $WORD_MODEL \
    --source-lang $LANG.word.src \
    --target-lang $LANG.word.tgt \
    --beam 5 \
    --alpha 1.5  \
    --batch-size 256 \
    --buffer-size 256 < "${NAME}.dev.src" | \
    grep -P '^H-'  | cut -c 3- | awk -F "\t" '{print $NF}' | \
    python postprocess_fairseq.py "word" > $NAME.dev.values

# build the dictionary
sed "s/ //g" $NAME.dev.src | paste - $NAME.dev.values > $NAME.dev.dict
cut -f 1 "${DATA}.dev.tsv" | python segment_table.py $NAME.dev.dict > $NAME.dev.pred

cut -f 1 "${DATA}.dev.tsv" | paste - $NAME.dev.pred > $NAME.guess

python 2022SegmentationST/evaluation/evaluate_word.py --gold $DATA.dev.tsv --guess $NAME.guess

# is it feasible to do this on a training set? I believe so. The English europarl
# set has 75k unique types (not too different from the task dev set. The
# Hungarian set has 350k types, which is significantly more but still not that
# bad (probably better than morfessor).
# fairseq-generate
# - the result of the model is bitext from raw->segmented. turn this bitext
#   into a dictionary.
# - apply that dictionary to the raw corpus
# (possibly separate: compute statistics about the resulting segmentation)

