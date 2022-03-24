WORD_MODEL="eng.word-entmax-512-1024-2-0.3-256-1.5/checkpoint_best.pt"

DATA="2022SegmentationST/data/eng.sentence"

# General idea:
# - get unique types from corpus (applying some kind of length filtering)
# fix these paths
# python unique_types.py < other-data/europarl/Europarl.en-hu.en > other-data/train.eng.words.src

cut -f 1 "${DATA}.dev.tsv" | python unique_types.py | sed 's/./& /g' > eng.sentence.dev.src

# - segment these types (this will require preprocessing them first)
fairseq-interactive \
    data-bin/eng.word \
    --path $WORD_MODEL \
    --source-lang eng.word.src \
    --target-lang eng.word.tgt \
    --beam 5 \
    --alpha 1.5  \
    --batch-size 256 \
    --buffer-size 256 < eng.sentence.dev.src | \
    grep -P '^H-'  | cut -c 3- | awk -F "\t" '{print $NF}' | \
    python postprocess_fairseq.py > eng.sentence.dev.values

# build the dictionary
sed "s/ //g" eng.sentence.dev.src | paste - eng.sentence.dev.values > eng.sentence.dev.dict
cut -f 1 "${DATA}.dev.tsv" | python segment_table.py eng.sentence.dev.dict > eng.sentence.dev.pred

cut -f 1 "${DATA}.dev.tsv" | paste -eng.sentence.dev.pred > eng.sentence.guess

python 2022SegmentationST/evaluation/evaluate_word.py --gold $DATA.dev.tsv --guess eng.sentence.guess

# is it feasible to do this on a training set? I believe so. The English europarl
# set has 75k unique types (not too different from the task dev set. The
# Hungarian set has 350k types, which is significantly more but still not that
# bad (probably better than morfessor).
# fairseq-generate
# - the result of the model is bitext from raw->segmented. turn this bitext
#   into a dictionary.
# - apply that dictionary to the raw corpus
# (possibly separate: compute statistics about the resulting segmentation)

