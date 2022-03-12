DATA=$1  # no "train.tsv" or "dev.tsv" suffix

NAME=$( basename $DATA )

# copy unsegmented column to a temporary file
TRAIN=$NAME.train.tmp
cut -f 1 $DATA.train.tsv > $TRAIN

morfessor-train -s $NAME.bin $TRAIN

# segment dev set
DEV=$NAME.dev.tmp
cut -f 1 $DATA.dev.tsv > $DEV
