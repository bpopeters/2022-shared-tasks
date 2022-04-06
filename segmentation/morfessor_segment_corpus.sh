# morfessor-segment obligatorily prints each word to a new line, which is
# problematic in the case that you want to segment a whole corpus (similarly to
# how you would with spm). So, this script runs morfessor-segment and
# postprocesses it so it is in lines corresponding to its input.

MODEL=$1
CORPUS=$2

morfessor-segment $CORPUS -l $MODEL | python label_suffixes.py > segments.tmp

python line_lengths.py < $CORPUS > lengths.tmp

# then, we need to rebuild sentences from that.
python rebuild_sentences.py lengths.tmp segments.tmp

rm {segments,lengths}.tmp
