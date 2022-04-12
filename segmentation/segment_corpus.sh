MODEL=$1
IN_CORPUS=$2
OUT_CORPUS=$3
NAME=$4  # e.g. eng.word
EXPNAME=$OUT_CORPUS

# General idea:
# - get unique types from corpus (applying some kind of length filtering)
# fix these paths
# python scripts/unique_types.py < other-data/europarl/Europarl.en-hu.en > other-data/train.eng.words.src
python scripts/unique_types.py 3 < $IN_CORPUS > $EXPNAME.uniqs.tmp

# - segment these types (this will require preprocessing them first)
fairseq-interactive \
    data-bin/$NAME \
    --path $MODEL \
    --source-lang $NAME.src \
    --target-lang $NAME.tgt \
    --beam 5 \
    --alpha 1.5  \
    --batch-size 256 \
    --buffer-size 256 < $EXPNAME.uniqs.tmp > $EXPNAME.uniqs.out
    cat $EXPNAME.uniqs.out | grep -P '^H-'  | cut -c 3- | awk -F "\t" '{print $NF}' > $EXPNAME.uniqs.wtf
    python scripts/postprocess_fairseq.py $NAME < $EXPNAME.uniqs.wtf > $EXPNAME.uniqs.segmented

# build the dictionary
sed "s/ //g" $EXPNAME.uniqs.tmp | paste - $EXPNAME.uniqs.segmented > $EXPNAME.segment_dict.tsv
python segment_table.py $EXPNAME.segment_dict.tsv < $IN_CORPUS > $OUT_CORPUS

# is it feasible to do this on a training set? I believe so. The English europarl
# set has 75k unique types (not too different from the task dev set. The
# Hungarian set has 350k types, which is significantly more but still not that
# bad (probably better than morfessor).
# fairseq-generate
# - the result of the model is bitext from raw->segmented. turn this bitext
#   into a dictionary.
# - apply that dictionary to the raw corpus
# (possibly separate: compute statistics about the resulting segmentation)

