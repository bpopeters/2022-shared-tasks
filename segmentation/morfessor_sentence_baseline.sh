DATA=$1  # no "train.tsv" or "dev.tsv" suffix

NAME=$( basename $DATA )

# copy unsegmented column to a temporary file
TRAIN=$NAME.train.mor
cut -f 1 $DATA.train.tsv > $TRAIN

morfessor-train -s $NAME.bin $TRAIN

# segment dev set
DEV=$NAME.dev.mor
cut -f 1 $DATA.dev.tsv > $DEV

# morfessor-segment is incredibly annoying because it outputs words on separate
# lines. But we want to keep the original structure of the dev file!
# First, we prepend @@ to every token except the first of each word
morfessor-segment $DEV -l $NAME.bin | python label_suffixes.py > segments.tmp

python line_lengths.py < $DEV > lengths.tmp

# then, we need to rebuild sentences from that.
python rebuild_sentences.py lengths.tmp segments.tmp > $NAME.mor.out

cut -f 1 $DATA.dev.tsv | paste - $NAME.mor.out > $NAME.mor.guess

python 2022SegmentationST/evaluation/evaluate_word.py --gold $DATA.dev.tsv --guess $NAME.mor.guess
