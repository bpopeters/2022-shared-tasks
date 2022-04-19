TRAIN=$1  # no "train.tsv" or "dev.tsv" suffix
DEV=$2

morfessor-train -s corpus-exp/morfessor.europarl.bin $TRAIN

bash morfessor_segment_corpus.sh corpus-exp/morfessor.europarl.bin $TRAIN > corpus-exp/europarl.mor-ep.en
bash morfessor_segment_corpus.sh corpus-exp/morfessor.europarl.bin $DEV > corpus-exp/flores.mor-ep.en
