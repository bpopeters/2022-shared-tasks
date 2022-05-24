DATA_BIN=$1
MODEL_PATH=$2
BEAM="${3}"
shift 3

MODEL_DIR=$( dirname $MODEL_PATH )


cat $DATA_BIN/dev.src | \
    fairseq-interactive \
        $DATA_BIN \
        --path $MODEL_PATH \
        --source-lang src \
        --target-lang tgt \
        --buffer-size 256 \
        --nbest $BEAM \
        --beam $BEAM \
        --unnormalized \
        "$@" | \
        grep -P '^H-' | python scripts/compute_beam_exactness.py \
        > "${MODEL_DIR}/dev-${BEAM}.exactness"

