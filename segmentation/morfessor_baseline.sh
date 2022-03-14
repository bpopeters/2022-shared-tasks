DATA=$1  # no "train.tsv" or "dev.tsv" suffix

NAME=$( basename $DATA )

# copy unsegmented column to a temporary file
TRAIN=$NAME.train.mor
cut -f 1 $DATA.train.tsv | python preprocess_morfessor.py > $TRAIN

morfessor-train -s $NAME.bin $TRAIN

# segment dev set
DEV=$NAME.dev.mor
cut -f 1 $DATA.dev.tsv | python preprocess_morfessor.py > $DEV

# problem with this is that it incorrectly handles entries that have a space in
# them. How do we handle that? One possibility is to replace spaces with "_"
morfessor-segment $DEV -l $NAME.bin | python postprocess_morfessor.py > $DEV.out
