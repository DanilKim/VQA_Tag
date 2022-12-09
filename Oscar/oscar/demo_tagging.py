from __future__ import absolute_import, division, print_function

from run_vqa import VQADataset, vqa_args_parser
from run_vqa import test as vqa_test
from run_captioning import CaptionTSVDataset, ic_args_parser, build_dataset, get_predict_file, restore_training_settings
from run_captioning import test as ic_test

import argparse
import glob
import logging
import os, io
import random, copy, time, json, pickle, csv
import base64

import sys
sys.path.insert(0, '.')

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import (Dataset, DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler
import cv2
import _pickle as cPickle
from tqdm import tqdm

from oscar.modeling.modeling_bert import ImageBertForSequenceClassification, BertForImageCaptioning
from transformers.pytorch_transformers import WEIGHTS_NAME, BertTokenizer, BertConfig

from oscar.utils.task_utils import (_truncate_seq_pair, convert_examples_to_features_vqa,
                        output_modes, processors)
from oscar.utils.tsv_file import TSVFile
from oscar.utils.tsv_file_ops import tsv_writer
from oscar.utils.misc import (mkdir, set_seed, 
        load_from_yaml_file, find_file_path_in_yaml)
from oscar.utils.caption_evaluate import (evaluate_on_coco_caption,
        evaluate_on_nocaps, ScstRewardCriterion)
from oscar.utils.cbs import ConstraintFilter, ConstraintBoxesReader
from oscar.utils.cbs import FiniteStateMachineBuilder

MODEL_CLASSES = {
    'bert': (BertConfig, ImageBertForSequenceClassification, BertTokenizer),
}

# Show the image in ipynb
from IPython.display import clear_output, Image, display
#import PIL.Image
from PIL import Image
from matplotlib import pyplot as plt

from visualize.visualize_tool import Visualizer

def showarray(a, fmt='jpeg'):
    a = np.uint8(np.clip(a, 0, 255))
    plt.imshow(a)
    plt.draw()
    plt.pause(1) # pause how many seconds
    plt.close()

##### Pre-difine paths & directories ####
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(CUR_DIR, 'datasets')
data_dir = os.path.join(DATA_ROOT, 'test_images')
img_dir = os.path.join(data_dir, 'images')
vg_vocab = os.path.join(DATA_ROOT, 'vg_objects_vocab_1600-400-20.txt')
ans2label_file = os.path.join(DATA_ROOT, 'vqa/cache/trainval_ans2label.pkl')
img_file_list = os.listdir(img_dir)
result_dir = os.path.join(data_dir, 'tagging_results')
if not os.path.isdir(result_dir):
    os.mkdir(result_dir)
vis_dir = os.path.join(result_dir, 'visualize')
if not os.path.isdir(vis_dir):
    os.mkdir(vis_dir)

################# Object Vocabulary ################
vg_classes = []
with open(vg_vocab, 'r') as f:
    for object in f.readlines():
        vg_classes.append(object.split(',')[0].lower().strip())


############ Default Catogories & Options ##########
DefaultCatList = 'car, shirt, pants, skirt, jacket, ' + \
                 'shoe, bag, laptop, phone, tv, sofa, ' + \
                 'chair, table, shelf, toy, dall'
DefaultMetaCat = 'car: car' + '\n' + \
                 'clothings: shirt, pants, skirt, jacket, shoe, bag' + '\n' + \
                 'electronics: laptop, phone, tv' + '\n' + \
                 'furniture: sofa, chair, table, shelf' + '\n' + \
                 'toy: toy, dall'
DefaultTagQuestions = \
                'general' + '\n' + \
                'kind: what kind of <CAT> is this?' + '\n' + \
                'color: what color is the <CAT>?' + '\n' + \
                'made_of: what is this <CAT> made of?' + '\n' + \
                'used: is this <CAT> been used?' + '\n' + \
                'purpose: what is the purpose of this <CAT>?' + '\n' + \
                'car' + '\n' + \
                'brand: which brand is this <CAT>?' + '\n' + \
                'fuel: is this <CAT> gasolin or disel or hybrid or electric?' + '\n' + \
                'clothings' + '\n' + \
                'size: what is the size of this <CAT>?' + '\n' + \
                'gender: is this <CAT> for man or woman or both?' + '\n' + \
                'design: what is the design of this <CAT>?' + '\n' + \
                'wash: in which season is this <CAT> for?' + '\n' + \
                'brand: what is the brand of this <CAT>?' + '\n' + \
                'electronics' + '\n' + \
                'brand: which company made this <CAT>?' + '\n' + \
                'weight: is this <CAT> heavy?' + '\n' + \
                'battery: is this <CAT> uses battery?' + '\n' + \
                'display: is the display of <CAT> LCD or OLED?' + '\n' + \
                'toy' + '\n' + \
                'figure: what does this <CAT> look like?' + '\n' + \
                'age: in which age can play with this <CAT>?' + '\n' + \
                'gender: is this <CAT> for boy or girl or both?' + '\n' + \
                'hardness: is this <CAT> hard or soft?' + '\n' + \
                'size: is this <CAT> big or small?' + '\n' + \
                'furniture' + '\n' + \
                'place: where to put this <CAT>?' + '\n' + \
                'weight: is this <CAT> heavy?' + '\n' + \
                'price: is this <CAT> cheap or expensive?'


################## Image Tagger Instance #####################
class CategoryTagger():
    def __init__(self, cat_list: str=None, metacat: str=None, tag_questions: str=None):
        self.cat_set = self.generate_categories(cat_list)
        self.cat_map = {cat: i for i, cat in enumerate(list(self.cat_set))}
        self.METACategories, self.Categories = self.generate_metacat(metacat)
        assert self.cat_set == set(self.Categories.keys())
        self.obj_ids = {cat: vg_classes.index(cat) for cat in list(self.cat_set)}
        self.tag_questions = self.generate_tag_questions(tag_questions)
        self.print_result()

    def generate_categories(self, cat_list: str=None) -> set:
        if cat_list is None:
            cat_list = input(
                "\nType the list of product categories to tag. \n" + \
                "Ex) >>> car, toy, jacket, skirt, shoe, pants, shirt \n>>> "
            )
        cat_list = cat_list.strip().lower().replace(',','').split()
        cat_set = set(cat_list).intersection(set(vg_classes))
        time.sleep(0.5)
        print(
            "\nAvailable tagging product categories are: \n %s" % ', '.join(list(cat_set))
        )
        time.sleep(0.5)
        return cat_set
    
    def generate_metacat(self, metacat_cat: str=None) -> (list, dict):
        if metacat_cat is None:
            return self.generate_user_metacat()
        else:
            METACategories = [mc.split(':')[0].strip() for mc in metacat_cat.split('\n')]
            Categories = {cat.strip().lower(): mc.split(':')[0].strip() \
                          for mc in metacat_cat.split('\n') \
                          for cat in mc.split(':')[1].split(',') \
                          if cat.strip().lower() in list(self.cat_set)}
            assert len(Categories.keys()) == len(set(Categories.keys()))
        
        return METACategories, Categories

    def generate_user_metacat(self) -> (list, dict):
        METACategories = []
        Categories = {}
        remain_set = self.cat_set.copy()
        while len(remain_set) > 0:
            print("\n** Remaining unmatched categories : [ %s ]" % ', '.join(list(remain_set)))
            time.sleep(0.5)
            input_list = input(
                "\nDefine a Meta-category and match with its sub-categories. \n" + \
                "Ex) >>> clothings: skirt, shoe, pants, shirt \n>>> "
            )
            time.sleep(0.3)
            try:
                meta_cat, sub_cats = input_list.split(':')
            except: 
                print("\nWrong input signal. Please retry.")
                time.sleep(0.3)
                continue
            if meta_cat in METACategories:
                print("\nAlready defined meta category: %s \n Try Again." % meta_cat)
                time.sleep(0.3)
            else:
                METACategories.append(meta_cat)
            sub_cats = sub_cats.split(',')
            sub_set = set([cat.strip().lower() for cat in sub_cats])
            inval_set = sub_set - self.cat_set
            alrdy_set = sub_set.intersection(self.cat_set) - remain_set
            sub_set = sub_set.intersection(remain_set)
            if len(inval_set) > 0:
                print("\nInvalid tagging categories: %s Dropped" % ', '.join(list(inval_set)))
                time.sleep(0.3)
                sub_set = sub_set - inval_set
            if len(alrdy_set) > 0:
                print("\nCategories: %s already matched, thus Dropped" % ', '.join(list(alrdy_set)))
                time.sleep(0.3)
                sub_set = sub_set - alrdy_set

            for cat in list(sub_set):
                Categories.update({cat: meta_cat})

            remain_set = remain_set - sub_set
        
        time.sleep(0.5)
        print("\nDONE matching categories!!\n")
        for meta_cat in METACategories:
            match_result = [cat for cat, meta in Categories.items() if meta == meta_cat]
            print("[ %s ] : %s" % (meta_cat, ', '.join(match_result)))
        print("\n")
        return METACategories, Categories

    def generate_tag_questions(self, tag_questions: str=None) -> dict:
        if tag_questions is None:
            return self.generate_user_questions()
        else:
            TAG_QUE = {}
            tag_questions = tag_questions.split('\n')
            for line in tag_questions:
                if len(line.split(' ')) < 4:
                    TAG_QUE[line] = []
                    assert line in self.METACategories + ['general']
                    meta = line
                else:
                    tag_op, que = line.split(':')
                    TAG_QUE[meta].append(
                        {"t": '#' + tag_op.strip().upper(),
                         "q": que.strip()}
                    )
            for meta in self.METACategories:
                if meta != 'general':
                    TAG_QUE[meta] = TAG_QUE['general'] + TAG_QUE[meta]
            return TAG_QUE
                

    def generate_user_questions(self) -> dict:
        TAG_QUE = {}
        TAG_QUE['general'] = []
        print(
            "\nDefine options to tag for \"all meta-categories\" in general, and create proper question" + \
            "\nBad question making might cause unwanted tagging result" + \
            "\n\nEx) >>> color: what color is this <CAT>?" + \
            "\n(The word '<CAT>' should be replaced with a specific category name later.)" + \
            "\nType [Q] if you are done."
        )
        while True:
            input_tag_que = input(
                ">>> "
            )
            if input_tag_que == 'Q':
                break
            try:
                tag_op, que = input_tag_que.split(':')
                TAG_QUE['general'].append({"t": '#' + tag_op.strip().upper(), 
                                           "q": que.strip()})
            except:
                print(
                    "\nWrong input signal. Please retry." + \
                    "\n\nEx) >>> color: what color is this <CAT>?" + \
                    "\n(The word '<CAT>' should be replaced with a specific category name later.)" + \
                    "\nType [Q] if you are done."
                )
        
        time.sleep(0.5)
        print("General Tagging Options & Questions for All Meta-Categories:")
        if len(TAG_QUE['general']) > 0:
            for item in TAG_QUE['general']:
                print(item["t"], item["q"])
        else:
            print("No general tagging option!")

        time.sleep(0.5)
        for meta in self.METACategories:
            TAG_QUE[meta] = TAG_QUE['general'][:]
            print(
                "\nDefine options to tag for a meta-category <%s>, and create proper question" % meta + \
                "\nBad question making might cause unwanted tagging result" + \
                "\n\nEx) >>> color: what color is this <CAT>?" + \
                "\n(The word '<CAT>' should be replaced with a specific category name later.)" + \
                "\nType [Q] if you are done."
            )
            while True:
                input_tag_que = input(
                    ">>> "
                )
                if input_tag_que == 'Q':
                    if len(TAG_QUE[meta]) == 0:
                        print("\nYou must add at least one tagging option & question")
                        print("\nTry again.\n")
                        continue
                    else:
                        break
                try:
                    tag_op, que = input_tag_que.split(':')
                except:
                    print("\nWrong input signal. Please retry.")
                    print("\nEx) >>> color: what color is this <CAT>?")
                    print("Type [Q] if you are done.")
                TAG_QUE[meta].append({"t": '#' + tag_op.strip().upper(), 
                                      "q": que.strip()})
            assert len(TAG_QUE[meta]) > 0
            time.sleep(0.5)

        return TAG_QUE

    def print_result(self):
        for meta in self.METACategories:
            print("\nCategories in <%s> : %s " % (meta, ', '.join(
                    [c for c, m in self.Categories.items() if m == meta]
                )))
            print("\nTagging options for meta-category <%s>:" % meta)
            for item in self.tag_questions[meta]:
                print(item["t"], item["q"])

############## User-define or Pre-defined ###############
while True:
    input_option = input(
        "Choose whether to define your own categories and tagging options (U), or use default (D).\n" + \
        "[U/D] >>> "
    )
    if input_option == 'U':
        image_tagger = CategoryTagger()
        break
    elif input_option == 'D':
        image_tagger = CategoryTagger(DefaultCatList, DefaultMetaCat, DefaultTagQuestions)
        break
    else:
        print("Wrong input. Try again.")

################### VQA, IC Options ####################
args = vqa_args_parser()
ic_args = ic_args_parser()

#################### Device Setting ####################
os.environ["CUDA_VISIBLE_DEVICES"]="4"
# Setup CUDA, GPU & distributed training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.n_gpu = torch.cuda.device_count()
args.device = device
ic_args.n_gpu = args.n_gpu
ic_args.device = device

# Set seed
set_seed(args.seed, args.n_gpu)

#########################################################
################### Image Captioning ####################
#########################################################
print("\nRun Captioning... ")
ic_args.do_test = True

config_class, model_class, tokenizer_class = BertConfig, BertForImageCaptioning, BertTokenizer
checkpoint = ic_args.eval_model_dir
assert os.path.isdir(checkpoint)
config = config_class.from_pretrained(checkpoint)
config.output_hidden_states = ic_args.output_hidden_states
tokenizer = tokenizer_class.from_pretrained(checkpoint)
model = model_class.from_pretrained(checkpoint, config=config)

model.to(args.device)
if args.n_gpu > 1:
    model = torch.nn.DataParallel(model)

ic_args = restore_training_settings(ic_args)
test_dataset = build_dataset(os.path.join(ic_args.data_dir, ic_args.test_yaml), 
                tokenizer, ic_args, is_train=False)
predict_file = os.path.join(result_dir, 'caption_result.tsv')
#predict_file = get_predict_file(checkpoint, test_dataset.yaml_file, ic_args)
cap_file = ic_test(ic_args, test_dataset, model, tokenizer, predict_file)

caption_tsv = TSVFile(cap_file, True)
img_id_2_idx = {caption_tsv.seek(i)[0] : i for i in range(caption_tsv.num_rows())}

model = None
test_dataset = None
print("Done!")
##########################################################
##################### VQA Tagging ########################
##########################################################

################### Prepare VQA Input ####################
print("\nPreparing VQA input...")
args.do_lower_case = True
args.do_test = True

feature_dict = torch.load(
    os.path.join(data_dir, 'test2015_img_frcnn_feats.pt')
)
with open(os.path.join(data_dir, 'test_images_d2obj_10_100_tags.json'), 'r') \
    as tag_file:
    tag_dict = json.load(tag_file)
with open(ans2label_file,'rb') \
    as a2l_file:
    ans2label = pickle.load(a2l_file)

sample_list = []
img_2_prod = {}
for img_fn in tqdm(img_file_list):
    img_id = img_fn.split('.')[0]

    # prepare_caption
    img_idx = img_id_2_idx[img_id]
    cap = json.loads(caption_tsv.seek(img_idx)[1])[0]['caption']

    tags = tag_dict[img_id]['tags'].split()
    prods = [cat for cat in image_tagger.Categories.keys() if cat in tags] \
      + [cat for cat in image_tagger.Categories.keys() if cat in cap.strip().split()]
    prods = list(set(prods))
    img_2_prod[img_id] = prods
    if prods == []: continue

    for category in prods:
        meta_cat = image_tagger.Categories[category]
        cat_id = image_tagger.cat_map[category]
        Questions = image_tagger.tag_questions[meta_cat]
        
        for qi, q in enumerate(Questions):
            sample_list.append({
                "q": q["q"].replace('<CAT>', category),
                "o": tag_dict[img_id]['tags'],
                "img_id": img_id,
                "q_id": '1%03d%07d%03d' % (cat_id, int(img_id), qi)
            })

with open(os.path.join(data_dir, 'test2015_qla_mrcnn.json'), "w") as qla_file:
    json.dump(sample_list, qla_file)

print("Done!")

################### VQA model forward ####################
print("\nRun VQA Tagging...")
processor = processors[args.task_name]()
args.output_mode = output_modes[args.task_name]
label_list = processor.get_labels(ans2label_file)
num_labels = len(label_list)

config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
config = config_class.from_pretrained(
    args.config_name if args.config_name else args.model_name_or_path,
    num_labels=num_labels, finetuning_task=args.task_name,
)
tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)

# discrete code
config.img_feature_dim = args.img_feature_dim
config.img_feature_type = args.img_feature_type
config.loss_type = args.loss_type[1][0]
config.classifier = args.classifier
config.cls_hidden_scale = args.cls_hidden_scale

model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config)

model.to(args.device)

test_dataset = VQADataset(args, 'test2015', tokenizer)

# Test
checkpoints = [args.output_dir]
if args.eval_all_checkpoints:
    checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))

for checkpoint in checkpoints:
    global_step = checkpoint.split('_')[-1] if len(checkpoints) > 1 else ""
    model = model_class.from_pretrained(checkpoint)
    model.to(args.device)
    vqa_test(args, model, test_dataset, prefix=global_step)

with open(data_dir + '/test2015_results.json', 'r') as fp:
    test_res = json.load(fp)

qid2ans = {res['question_id']:res['answer'] for res in test_res}
print("Done!")

##########################################################
############ Save & Visualize Tagging Results ############
##########################################################
while True:
    vis = input("Do you want to also save visualization result? [Y/N] ")
    if vis == 'Y':
        print("\nSaving Tagging Results w. Visualization...")
        vis = True; break
    elif vis =='N':
        print("\nSaving Tagging Results w/o Visualization...")
        vis = False; break
    else:
        print("Wrong input\n")

with open(os.path.join(result_dir, 'tagging_results.tsv'), 'wt') as out_file:
    res_writer = csv.writer(out_file, delimiter='\t')
    res_writer.writerow(['image', 'product', 'option', 'tag'])
    for img_fn in tqdm(img_file_list):
        im = cv2.imread(os.path.join(img_dir, img_fn))

        # category info of an image
        img_id = img_fn.split('.')[0]
        tags = tag_dict[img_id]['tags'].split()

        # OD info of an image
        objects_id = torch.Tensor(tag_dict[img_id]['objects_id'])
        objects_conf = torch.Tensor(tag_dict[img_id]['objects_conf'])
        objects_box = torch.Tensor(tag_dict[img_id]['boxes'])
        
        # prepare visualizing image
        v = Visualizer(im[:, :, :], scale=1.2)
        max_boxes = []
        out_txts = []

        # prepare_caption
        img_idx = img_id_2_idx[img_id]
        cap = json.loads(caption_tsv.seek(img_idx)[1])[0]['caption']

        for category in img_2_prod[img_id]:
            if vis:
                # draw highest scoring object box on image
                obj_id = image_tagger.obj_ids[category]
                cat_objs = (objects_id == obj_id).nonzero(as_tuple=True)[0]
                if len(cat_objs) == 0:
                    max_boxes.append(torch.Tensor([[0, 0, 0.1, 0.1]]))
                else:
                    cat_scores = objects_conf[cat_objs]
                    cat_boxes = objects_box[cat_objs]
                    max_boxes.append(cat_boxes[cat_scores.argmax()].unsqueeze(0))

            # Visualize Q&A
            meta_cat = image_tagger.Categories[category]
            cat_id = image_tagger.cat_map[category]
            Questions = image_tagger.tag_questions[meta_cat]
            #print('< ' + meta_cat + ': ' + category + ' >')
            out_txt = '<%s>\n' % category
            for qi, q in enumerate(Questions):
                qid = int('1%03d%07d%03d' % (cat_id, int(img_id), qi))
                res_writer.writerow([img_fn, category, q["t"], qid2ans[qid]])
                out_txt += "%s : %s \n" % (q["t"], qid2ans[qid])
            out_txts.append(out_txt)

        if vis:
            if max_boxes == []:
                ingredient = {'box': None, 'text': None}
            else:
                max_boxes = torch.cat(max_boxes, 0)
                ingredient = {'box': max_boxes, 'text': out_txts}
            v = v.draw_instance_predictions(ingredient)

            result_path = os.path.join(result_dir, 'visualize', os.path.basename(img_fn))
            cv2.imwrite(result_path, v.get_image())

print("Done!")
os.remove(caption_tsv.lineidx)
print("\nAll Done!!")


