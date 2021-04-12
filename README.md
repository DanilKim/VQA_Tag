# User-Interactive Automatic Image Tagging via Visual Question Answering  <img src="docs/oscar_logo.png" width="200" align="right"> 

## Introduction
This repository contains source code and instructions for automatic image tagging using Visual Question Answering (VQA) system. 
We utilized the VQA model architecture and trained parameters introduced in [Oscar: Object-Semantics Aligned Pre-training for Vision-Language Tasks](https://arxiv.org/abs/2004.06165) and Oscar team's shared [code base](https://github.com/microsoft/Oscar)  
They proposed a cross-modal pre-training method **Oscar** (Object-Semantics Aligned Pre-training), which leverages **object tags** detected in images as anchor points to significantly ease the learning of image-text alignments. We pre-train Oscar on the public corpus of 6.5 million text-image pairs, and fine-tune it on downstream tasks, creating new state-of-the-arts on six well-established vision-language understanding and generation tasks. For more on this project, see the [Microsoft Research Blog post](https://www.microsoft.com/en-us/research/blog/objects-are-the-secret-key-to-revealing-the-world-between-vision-and-language/).

This automatic image tagging system allows users to gather free kinds of target images containing diverse objects, and to decide what kind of objects and tagging options are preferred.
This tagging system can be particulary useful for **e-commerse**, to gather product meta-data for enlarging product searching database with image.
Other various applications can be also available. 

<img src="docs/oscar.PNG" width="650"> 

## Sutup and Install
Move to your working directory, then clone from repository.
```console
git clone <repository for VQATag>
cd VQA_Tag
```

### Environments
First install docker on your computer following [Docker Installation Guide](https://docs.docker.com/engine/install/ubuntu/), if not installed.
Pull seperate docker images for running feature extraction & VQA model respectively.
```console
docker pull onlyone21/detectron2:pytorch1.7.1-cuda11.1
docker pull onlyone21/oscar:pytorch1.7.1-cuda11.0-cudnn8-devel 
```

### File System
Download data to designated directories.
```console
wget https://biglmdiag.blob.core.windows.net/oscar/datasets/vqa.zip
unzip vqa.zip -d Oscar/oscar/datasets/vqa

wget https://biglmdiag.blob.core.windows.net/oscar/datasets/coco_caption.zip
unzip vqa.zip -d Oscar/oscar/datasets/coco_caption

wget https://biglmdiag.blob.core.windows.net/oscar/exp/vqa/base/vqa_base_best.zip
unzip vqa_base_best.zip -d Oscar/oscar/model_ckpts/vqa
mv Oscar/oscar/model_ckpts/vqa Oscar/oscar/modelckpts/vqa_base_best

wget https://biglmdiag.blob.core.windows.net/oscar/exp/coco_caption/base/checkpoint.zip
unzip checkpoint.zip -d Oscar/oscar/model_ckpts/image_captioning
```
The file tree should be set as following tree:
```
VQA_Tag
  |
  |- image_feature_extraction.sh
  |- user_defined_image_tagging.sh
  |- clean_up.sh
	|
  |- py-bottom-up-attention
  |    |- ...
  |   ...
  |- Oscar
  |    |- oscar
	|    |    |- datasets
  |    |    |     |- test_images
	|    |    |     |   |- test.yaml
  |    |    |     |- coco_caption
  |    |    |     |- vqa
	|    |    |- model_ckpts
  |    |    |     |- image_captioning
	|    |    |     |   |- checkpoint-29-66420
  |    |    |     |- vqa
	|    |    |     |   |- vqa_base_best 
  |    |   ...
	|    |- transformer
	|    |- coco_caption
  |   ...   
	|
  |- test_images  <-- Add your target images in 'test_images' folder (jpg files)
       |- 1.jpg
       |- 2.jpg
      ...
       |- 10000.jpg
```
Add your target images in **VQA_Tag/test_images** folder (jpg files)
> *Each image file must have a unique positive integer name. (1~9999999.jpg)*

**clean_up.sh** cleans up all intermediate outputs, keeping target images & tagging results untouched.

## Instruction
### Feature Extraction
Extract object region, tag and features through [Object Detector](https://github.com/airsplay/py-bottom-up-attention)
```console
docker run -it --rm --ipc host --gpus all -v <path_to_VQATag>:/root -w /root --name detectron2-1.7 onlyone21/detectron2:pytorch1.7.1-cuda11.1 /bin/bash
bash image_feature_extraction.sh
```
<img src="docs/feature_extraction.png" width="650">

Exit docker environment
```console
exit
```

### VQA Tagging
User defines objects and tagging options, or use defualt. 
Then, implements tagging via VQA model.
```console
docker run -it --rm --ipc host --gpus all -v <path_to_VQATag>:/root -w /root --name oscar onlyone21/oscar:pytorch1.7.1-cuda11.0-cudnn8-devel /bin/bash
bash user_defined_image_tagging.sh
```

- User input required (U/D) whether to define product & options (U), or use pre-defined defualt settings (D)
  <img src="docs/start.png" width="650">
  - D : Default settings
  - U : User-Defined
    1. Define tagging object categories. (**User Input**)
      (CAUTION: The system is sensitive to object word selection. shoes -> shoe, shirts -> shirt, television -> tv)
      <img src="docs/categories_input.png" width="400">
      Unavailable input categories (Not in [OD dictionary](https://github.com/airsplay/py-bottom-up-attention/blob/master/demo/data/genome/1600-400-20/objects_vocab.txt)) are removed.
      <img src="docs/categories_result.png" width="400">

    2. Define meta-categories, and match each meta-category with its sub-categories. (**User Input**)
      <img src="docs/metacategories_input.png" width="650">
      Matching Result:
      <img src="docs/metacategories_result.png" width="300">
    
    3. For each meta-catogory, define tagging options and corresponding questions
      1. Define 'general' tagging options and questions applied to all meta-categories in common. (**User Input**)
        <img src="docs/general_options_questions_input.png" width="650">
      2. Define 'specific' tagging options and questions for each meta-category. (**User Input**)
        (If No input for a meta-category, general options & questions are applied)
        <img src="docs/specific_options_questions_input.png" width="650">

  - Input results summerized:
    <img src="docs/summerized_input_result.png" width="400">
- Image Captioning 
  <img src="docs/image_captioning.png" width="650">
- VQA Tagging
  <img src="docs/vqa_tagging.png" width="650">
- Decide whether to save visualized results
  <img src="docs/save_visualize.png" width="650">

```console
exit
```

## Results
- tagging_results/caption_result.tsv
  <img src="docs/caption_result.png" width="650">
- tagging_results/tagging_results.tsv
  <img src="docs/tagging_results.png" width="400">
- tagging_results/images/16.jpg
  <img src="docs/visualize_result.png" width="650">


