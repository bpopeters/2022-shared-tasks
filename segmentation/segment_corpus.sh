MODEL=$1
IN_CORPUS=$2
OUT_CORPUS=$3
NAME=$OUT_CORPUS

# General idea:
# - get unique types from corpus (applying some kind of length filtering)
# fix these paths
# python unique_types.py < other-data/europarl/Europarl.en-hu.en > other-data/train.eng.words.src
python unique_types.py < $IN_CORPUS > $NAME.uniqs.tmp

# - segment these types (this will require preprocessing them first)
fairseq-interactive \
    data-bin/eng.word \
    --path $MODEL \
    --source-lang eng.word.src \
    --target-lang eng.word.tgt \
    --beam 5 \
    --alpha 1.5  \
    --batch-size 256 \
    --buffer-size 256 < $NAME.uniqs.tmp | \
    grep -P '^H-'  | cut -c 3- | awk -F "\t" '{print $NF}' | \
    python postprocess_fairseq.py > $NAME.uniqs.segmented

# build the dictionary
sed "s/ //g" $NAME.uniqs.tmp | paste - $NAME.uniqs.segmented > $NAME.segment_dict.tsv
python segment_table.py $NAME.segment_dict.tsv < $IN_CORPUS > $OUT_CORPUS

# is it feasible to do this on a training set? I believe so. The English europarl
# set has 75k unique types (not too different from the task dev set. The
# Hungarian set has 350k types, which is significantly more but still not that
# bad (probably better than morfessor).
# fairseq-generate
# - the result of the model is bitext from raw->segmented. turn this bitext
#   into a dictionary.
# - apply that dictionary to the raw corpus
# (possibly separate: compute statistics about the resulting segmentation)

