''' Computes CIDEr between generated sentences and classes'''

import numpy as np
import sys
sys.path.append('utils/')
from init import *
from eval import eval_generation
import pdb
import argparse
import os
import time
import pickle as pkl

sys.path.insert(0,coco_eval_path)
from pycocoevalcap.cider import cider
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

def compute_cider(gen, gt, tfidf_gt=None):
  cider_scorer = cider.Cider() 
  score, scores = cider_scorer.compute_score(gt, gen, tfidf_gt) 
  return scores, cider_scorer.imgIds
   
def eval_class_meteor(tag):
  #Make the reference for an image all reference sentences from the corresponding class.  

  gen_annotations = 'generated_sentences//birds_fg_test/gve_models/%s/beam1/generation_result.json' %tag

  if not os.path.isdir('cider_scores'):
    os.mkdir('cider_scores')

  dataset = 'birds_fg'
  split = 'test'
  gt_comp = 'train_noCub' #alternatively could use sentences from val or test. Can think of this metric as a measure between a generated sentence and the NN class in the reference set.  

  image_root = eval_generation.determine_image_pattern(dataset, split) 
  vocab_file = 'data/vocab.txt'
  vocab = open_txt(vocab_file)

  #combine gt annotations for each class
  anno_path_ref = eval_generation.determine_anno_path(dataset, gt_comp)
  ref_annotations = read_json(anno_path_ref)
  gen_annotations = read_json(gen_annotations)
  
  #create tfidf dict
  tfidf_dict = {} 
  for a in ref_annotations['annotations']:
    im = a['image_id']
    if im not in tfidf_dict.keys():
      tfidf_dict[im] = []
    tfidf_dict[im].append({'caption': a['caption'], 'id': a['image_id'], 'image_id': a['image_id']})

  #create dict which has all annotations which correspond to a certain class in the reference set
  gt_class_annotations = {}
  for a in ref_annotations['annotations']:
    cl = int(a['image_id'].split('/')[0].split('.')[0]) - 1 
    if cl not in gt_class_annotations:
      gt_class_annotations[cl] = {}
      gt_class_annotations[cl]['all_images'] = [] 
    gt_class_annotations[cl]['all_images'].append({'caption': a['caption'], 'id': a['image_id'], 'image_id': a['image_id']})
  
  #create dict which includes 200 different "test" sets with images only from certain classes
  gen_class_annotations = {}
  for a in gen_annotations:
    cl = int(a['image_id'].split('/')[0].split('.')[0]) - 1 
    im = a['image_id']
    if cl not in gen_class_annotations:
      gen_class_annotations[cl] = {}
    if im not in gen_class_annotations[cl].keys():
      gen_class_annotations[cl][im] = []
    gen_class_annotations[cl][im].append({'caption': a['caption'], 'id': a['image_id'], 'image_id': a['image_id']})

  #for tokenizer need dict with list of dicts
  t = time.time()
  tokenizer = PTBTokenizer()
  tfidf_dict = tokenizer.tokenize(tfidf_dict)
  for key in gt_class_annotations:
    gt_class_annotations[key] = tokenizer.tokenize(gt_class_annotations[key])
  for key in gen_class_annotations:
    gen_class_annotations[key] = tokenizer.tokenize(gen_class_annotations[key])
  print "Time for tokenization: %f." %(time.time()-t)

  score_dict = {}
  for cl in sorted(gen_class_annotations.keys()):
    gts = {}
    gen = {}
    t = time.time()
    for cl_gt in sorted(gt_class_annotations.keys()):
      for im in sorted(gen_class_annotations[cl].keys()):
        gen[im+('_%d' %cl_gt)] = gen_class_annotations[cl][im]
        gts[im+('_%d' %cl_gt)] = gt_class_annotations[cl_gt]['all_images']
    scores, im_ids = compute_cider(gen, gts) 
    for s, ii in zip(scores, im_ids):
      score_dict[ii] = s
    print "Class %s took %f s." %(cl, time.time() -t)


  pkl.dump(score_dict, open('cider_scores/cider_score_dict_%s.p' %(tag), 'w'))

if __name__ == '__main__':
  
  #tag indicates which sentences to evaluate.  Sentences are generated when running my eval code. 
  parser = argparse.ArgumentParser()
  parser.add_argument("--tag", type=str, default='None')
  args = parser.parse_args()

  eval_class_meteor(args.tag)

