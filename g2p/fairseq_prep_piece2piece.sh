readonly DATA_PATH=$1  # example: 2022-shared-tasks/data/eng.word
readonly OUT_PATH=$2
readonly VOCAB=$3
shift 3

bin() {
    tail -n +4 "${OUT_PATH}/src.vocab" | cut -f 1 | sed "s/$/ 100/g" > "${OUT_PATH}/src.fairseq.vocab"
    tail -n +4 "${OUT_PATH}/tgt.vocab" | cut -f 1 | sed "s/$/ 100/g" > "${OUT_PATH}/tgt.fairseq.vocab"
    # todo: fix testpref when it is available
    fairseq-preprocess \
        --source-lang="src" \
        --target-lang="tgt" \
        --trainpref="${OUT_PATH}/train" \
        --validpref="${OUT_PATH}/dev" \
        --testpref="${OUT_PATH}/test" \
        --tokenizer=space \
        --thresholdsrc=1 \
        --thresholdtgt=1 \
        --srcdict "${OUT_PATH}/src.fairseq.vocab" \
        --tgtdict "${OUT_PATH}/tgt.fairseq.vocab" \
        --destdir="${OUT_PATH}"
}

python tokenize.py "${DATA_PATH}_train.tsv" --src-tok-type spm --tgt-tok-type spm --vocab-size $VOCAB --out-dir $OUT_PATH --split train $@
python tokenize.py "${DATA_PATH}_dev.tsv" --src-tok-type spm --tgt-tok-type spm --vocab-size $VOCAB --existing-src-spm "${OUT_PATH}/src" --existing-tgt-spm "${OUT_PATH}/tgt" --out-dir $OUT_PATH --split dev
python tokenize.py "${DATA_PATH}_test.tsv" --src-tok-type spm --tgt-tok-type spm --vocab-size $VOCAB --existing-src-spm "${OUT_PATH}/src" --existing-tgt-spm "${OUT_PATH}/tgt" --out-dir $OUT_PATH --split test
bin

