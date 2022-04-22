

WORD_MODEL=$1
DATA=$2
NAME=$( basename $DATA )
LANG=$( basename $DATA .sentence)

# General idea:
# - get unique types from corpus (applying some kind of length filtering)
# fix these paths
# python scripts/unique_types.py < other-data/europarl/Europarl.en-hu.en > other-data/train.eng.words.src

# unique_types.py is currently problematic because it performs character-level
# tokenization
# but we can't quite use tokenize.py either because it does too much.

# get unique types from sentence-level dev set (todo: add option for whitespace/nltk pretokenization)
cut -f 1 "${DATA}.dev.tsv" | python scripts/unique_types.py > $NAME.dev.src

# apply whitespace, spm tokenization to input (how to do this depends on the
# contents of the word-level data-bin directory; if there is a sentencepiece
# model, use it; otherwise, back off to characters).

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
    python scripts/postprocess_fairseq.py "word" > $NAME.dev.values

# build the dictionary
sed "s/ //g" $NAME.dev.src | paste - $NAME.dev.values > $NAME.dev.dict
cut -f 1 "${DATA}.dev.tsv" | python segment_table.py $NAME.dev.dict > $NAME.dev.pred

cut -f 1 "${DATA}.dev.tsv" | paste - $NAME.dev.pred > $NAME.guess

python 2022SegmentationST/evaluation/evaluate.py --gold $DATA.dev.tsv --guess $NAME.guess

# is it feasible to do this on a training set? I believe so. The English europarl
# set has 75k unique types (not too different from the task dev set. The
# Hungarian set has 350k types, which is significantly more but still not that
# bad (probably better than morfessor).
# fairseq-generate
# - the result of the model is bitext from raw->segmented. turn this bitext
#   into a dictionary.
# - apply that dictionary to the raw corpus
# (possibly separate: compute statistics about the resulting segmentation)

