readonly DATA_PATH=$1  # example: 2022-shared-tasks/data/eng.word
readonly OUT_PATH=$2

bin() {
    tail -n +4 "${OUT_PATH}/src.vocab" | cut -f 1 | sed "s/$/ 100/g" > "${OUT_PATH}/src.fairseq.vocab"
    tail -n +4 "${OUT_PATH}/tgt.vocab" | cut -f 1 | sed "s/$/ 100/g" > "${OUT_PATH}/tgt.fairseq.vocab"
    # todo: fix testpref when it is available
    fairseq-preprocess \
        --source-lang="src" \
        --target-lang="tgt" \
        --trainpref="${OUT_PATH}/train" \
        --validpref="${OUT_PATH}/dev" \
        --testpref="${OUT_PATH}/dev" \
        --tokenizer=space \
        --thresholdsrc=1 \
        --thresholdtgt=1 \
        --srcdict "${OUT_PATH}/src.fairseq.vocab" \
        --tgtdict "${OUT_PATH}/tgt.fairseq.vocab" \
        --destdir="${OUT_PATH}"
}

python tokenize.py "${DATA_PATH}.train.tsv" --src-tok-type char --tgt-tok-type char --out-dir $OUT_PATH --split train
python tokenize.py "${DATA_PATH}.dev.tsv" --src-tok-type char --tgt-tok-type char --out-dir $OUT_PATH --split dev
bin

