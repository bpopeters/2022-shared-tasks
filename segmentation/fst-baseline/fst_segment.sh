# What do we need to have in order to do this?
# 1) The model name, e.g. ces.word
# The model n-gram order

ORDER=$1
DATA=$2
# DATA="../2022SegmentationST/data/ces.word"
NAME=$( basename $DATA)

PREFIX=$3

DEV_TSV="${DATA}.dev.tsv"
TEST_TSV=DEV_TSV
LANGUAGE=$( basename $DATA )
# Output symbols.
# need to fix this by postprocessing column 2 (see the other tasks we have)
# but also, what do we do about column 1?

cut -f 1 "${DEV_TSV}" > $PREFIX/dev.$NAME.words
python tok.py < $PREFIX/dev.$NAME.words > $PREFIX/dev.$NAME.splitwords
cut -f 2 "${DEV_TSV}" | python tok.py > $PREFIX/dev.$NAME.segments

paste $PREFIX/dev.$NAME.splitwords $PREFIX/dev.$NAME.segments > $PREFIX/dev.$NAME.tsv

echo "building ${ORDER}"
bash model \
    --encoder_path "${PREFIX}/model.enc" \
    --far_path "${PREFIX}/model.far" \
    --fst_path "${PREFIX}/${NAME}-${ORDER}.fst" \
    --order $ORDER
    
echo "predicting ${ORDER}"
bash predict \
    --input_path $PREFIX/dev.$NAME.tsv \
    --fst_path $PREFIX/$NAME-$ORDER.fst \
    --output_token_type "${PREFIX}/chars.sym" \
    --output_path "${PREFIX}/$NAME-$ORDER.pred"

# now, postprocess to make a guess file
sed "s/ //g" $PREFIX/$NAME-$ORDER.pred | sed "s/|/ @@/g" | paste $PREFIX/dev.$NAME.words - > $PREFIX/dev.$NAME-$ORDER.guess
