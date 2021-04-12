cp -r test_images py-bottom-up-attention/demo/data/images/
cd py-bottom-up-attention

CUDA_VISIBLE_DEVICES=3 python3 demo/feature_extractor_vg.py --split test_images

INIT="demo/data/features/test_images"
DEST="../Oscar/oscar/datasets/test_images"

mv ${INIT}/test_images_10_100_feat.tsv ${DEST}/test.feature.tsv
mv ${INIT}/test_images_10_100_tags.tsv ${DEST}/test.label.tsv
mv ${INIT}/test_images_img_frcnn_feats.pt ${DEST}/test2015_img_frcnn_feats.pt
mv ${INIT}/test_images_d2obj_10_100_tags.json ${DEST}/

rm -rf demo/data/images/test_images

cd ../

