CORPUS=$1
VOCAB=$2

spm_train --input $CORPUS --model_prefix corpus-exp/spm.europarl.en.$VOCAB --vocab_size $VOCAB --character_coverage 1.0
