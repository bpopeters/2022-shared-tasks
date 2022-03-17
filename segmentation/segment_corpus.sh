NAME="eng.word"

# General idea:
# - get unique types from corpus (applying some kind of length filtering)
# fix these paths
# python unique_types.py < other-data/europarl/Europarl.en-hu.en > other-data/train.eng.words.src
python unique_types.py < other-data/flores/eng.dev > other-data/dev.eng

# - segment these types (this will require preprocessing them first)
fairseq-interactive \
    data-bin/eng.word \
    --path fairseq-checkpoints/eng.word-entmax-512-512-2-.3-1.5/checkpoint_last.pt \
    --source-lang eng.word.src \
    --target-lang eng.word.tgt \
    --beam 5 \
    --alpha 1.5  \
    --batch-size 256 \
    --buffer-size 256 < other-data/dev.eng | \
    grep -P '^H-'  | cut -c 3- | awk -F "\t" '{print $NF}' | \
    python postprocess_fairseq.py > other-data/dev.eng.segmented

# build the dictionary
paste other-data/dev.eng other-data/dev.eng.segmented > other-data/segment_dict.tsv
python segment_table.py other/data_segment_dict.tsv < other-data/flores/eng.dev > other-data/flores.eng.segmented

# is it feasible to do this on a training set? I believe so. The English europarl
# set has 75k unique types (not too different from the task dev set. The
# Hungarian set has 350k types, which is significantly more but still not that
# bad (probably better than morfessor).
# fairseq-generate
# - the result of the model is bitext from raw->segmented. turn this bitext
#   into a dictionary.
# - apply that dictionary to the raw corpus
# (possibly separate: compute statistics about the resulting segmentation)

