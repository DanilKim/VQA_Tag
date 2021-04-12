cp -r test_images Oscar/oscar/datasets/test_images/images
cd Oscar

python oscar/demo_tagging.py
rm -rf oscar/datasets/test_images/images

cd ..
