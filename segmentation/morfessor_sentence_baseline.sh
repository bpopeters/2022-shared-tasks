DATA=$1  # no "train.tsv" or "dev.tsv" suffix

NAME=$( basename $DATA )

# copy unsegmented column to a temporary file
TRAIN=$NAME.train.mor
cut -f 1 $DATA.train.tsv > $TRAIN

morfessor-train -s $NAME.bin $TRAIN

# segment dev set
DEV=$NAME.dev.mor
cut -f 1 $DATA.dev.tsv > $DEV

bash morfessor_segment_corpus.sh $NAME.bin $DEV > $NAME.mor.out

paste $DEV $NAME.mor.out > $NAME.mor.guess

python 2022SegmentationST/evaluation/evaluate.py --gold $DATA.dev.tsv --guess $NAME.mor.guess
