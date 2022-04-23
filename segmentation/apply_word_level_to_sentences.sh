WORD_MODEL=$1
WORD_DATA_BIN=$2
SENTENCE_DEV_SET=$3
OUT_DIR=$4

mkdir -p $OUT_DIR

echo "${WORD_MODEL}\n${SENTENCE_DEV_SET}" > "${OUT_DIR}/description.txt"

# General idea:
# - get unique types from corpus (applying some kind of length filtering)
# fix these paths
# python scripts/unique_types.py < other-data/europarl/Europarl.en-hu.en > other-data/train.eng.words.src

# unique_types.py is currently problematic because it performs character-level
# tokenization
# but we can't quite use tokenize.py either because it does too much.

# get unique types from sentence-level dev set (todo: add option for whitespace/nltk pretokenization)
# in its current form, this will only work for char2*, but that
# should be ok for now because we don't have any word-level piece2piece models
cut -f 1 "${SENTENCE_DEV_SET}" | python scripts/unique_types.py > "${OUT_DIR}/sentence.uniq.src"

# apply whitespace, spm tokenization to input (how to do this depends on the
# contents of the word-level data-bin directory; if there is a sentencepiece
# model, use it; otherwise, back off to characters).

# - segment these types (this will require preprocessing them first)
fairseq-interactive \
    $WORD_DATA_BIN \
    --path $WORD_MODEL \
    --source-lang src \
    --target-lang tgt \
    --beam 5 \
    --alpha 1.5  \
    --batch-size 256 \
    --buffer-size 256 < "${OUT_DIR}/sentence.uniq.src" | \
    grep -P '^H-'  | cut -c 3- | awk -F "\t" '{print $NF}' | \
    python scripts/postprocess_fairseq.py "word" > "${OUT_DIR}/sentence.uniq.pred"

# build the dictionary
sed "s/ //g" "${OUT_DIR}/sentence.uniq.src" | paste - "${OUT_DIR}/sentence.uniq.pred" > "${OUT_DIR}/sentence.uniq.tsv"
cut -f 1 "${SENTENCE_DEV_SET}" | python segment_table.py "${OUT_DIR}/sentence.uniq.tsv" > "${OUT_DIR}/dev.pred"

cut -f 1 "${SENTENCE_DEV_SET}" | paste - "${OUT_DIR}/dev.pred" > "${OUT_DIR}/dev.guess"

python 2022SegmentationST/evaluation/evaluate.py --gold "${SENTENCE_DEV_SET}" --guess "${OUT_DIR}/dev.guess" > "${OUT_DIR}/dev.results"

# is it feasible to do this on a training set? I believe so. The English europarl
# set has 75k unique types (not too different from the task dev set. The
# Hungarian set has 350k types, which is significantly more but still not that
# bad (probably better than morfessor).
# fairseq-generate
# - the result of the model is bitext from raw->segmented. turn this bitext
#   into a dictionary.
# - apply that dictionary to the raw corpus
# (possibly separate: compute statistics about the resulting segmentation)

