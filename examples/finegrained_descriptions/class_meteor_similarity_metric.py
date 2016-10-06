import numpy as np
from init import *
from eval import eval_generation
import pdb
import argparse
import os
import time
import pickle as pkl

COCO_EVAL_PATH = '../../data/coco/coco-caption-eval/'
sys.path.insert(0,COCO_EVAL_PATH)
from pycocoevalcap.cider import cider
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

#def compute_cider(cider_scorer, gen, gt, tfidf_gt=None):
def compute_cider(gen, gt, tfidf_gt=None):
  #cider gts are dict with id and all gt caps, gen are dict with id and one generated sentence
  #probably don't need loop; can do all at once (?)
  cider_scorer = cider.Cider() 
  score, scores = cider_scorer.compute_score(gt, gen, tfidf_gt) 
  return scores, cider_scorer.imgIds
   
def eval_class_meteor(tag):
  gen_annotations = 'generated_sentences//birds_fg_test/snapshots/%s/beam1/generation_result.json' %tag
  #tag = 'nearest_neighbor_baseline'
  #gen_annotations = 'generated_sentences/%s.json' %tag

  dataset = 'birds_fg'
  split = 'test'

  image_root = eval_generation.determine_image_pattern(dataset, split) 
  vocab_file = '%s/%s.txt' %(eval_generation.determine_vocab_folder(dataset, split), 'CUB_vocab_noUNK')
  vocab = open_txt(vocab_file)

  #combine gt annotations for each class
  anno_path_train = eval_generation.determine_anno_path(dataset, 'train_noCub')
  #anno_path_train = eval_generation.determine_anno_path(dataset, 'val')
  train_annotations = read_json(anno_path_train)
  gen_annotations = read_json(gen_annotations)
  
  #create tfidf dict
  tfidf_dict = {} 
  for a in train_annotations['annotations']:
    im = a['image_id']
    if im not in tfidf_dict.keys():
      tfidf_dict[im] = []
    tfidf_dict[im].append({'caption': a['caption'], 'id': a['image_id'], 'image_id': a['image_id']})

  #create dict which has all annotations which correspond to a certain class in the train set
  gt_class_annotations = {}
  for a in train_annotations['annotations']:
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
  print "Time for tokenization: %f." %(t-time.time())

  #make ginormous cider dataset
#  gts = {}
#  gen = {}
#  t = time.time()
#  for cl_gt in sorted(gt_class_annotations.keys())[:10]:
#    for cl in sorted(gen_class_annotations.keys())[:5]:
#      for im in sorted(gen_class_annotations[cl].keys())[:10]:
#        gen[im+('_%d' %cl_gt)] = gen_class_annotations[cl][im]
#        gts[im+('_%d' %cl_gt)] = gt_class_annotations[cl_gt]['all_images']
#  print "Time to make giant dict: %f s" %(time.time() - t) 
#
#  scores, im_ids = compute_cider(gen, gts) 
#  score_dict = {}
#  for s, ii in zip(scores, im_ids):
#    score_dict[ii] = s
#  pkl.dump(score_dict, open('cider_scores/cider_score_dict_%sCHECK.p' %tag, 'w'))

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
    #pdb.set_trace()
    print "Class %s took %f s." %(cl, time.time() -t)

  pkl.dump(score_dict, open('cider_scores/cider_score_dict_%s_trainRef.p' %tag, 'w'))




#  cider_matrix = np.zeros((200,200))
#
#  cider_scorer = cider.Cider() 
#
#  for gt_cl in gt_class_annotations.keys(): #loop over all classes and compute meteor for each class
#    for gen_cl in gen_class_annotations.keys():
#
#      t = time.time()
#      class_gt = gt_class_annotations[gt_cl]
#      class_gen = gen_class_annotations[gen_cl]
#  
#      #make new ground truth dict for test caps
#      new_gt_annotations = {} #new_gt annotations will just be all train annotations that correspond to a given class
#
#      test_images = class_gen.keys()
#      count = 0
#      for im in test_images:
#        #add annotation
#        cl = int(im.split('/')[0].split('.')[0]) - 1
#        new_gt_annotations[im] = class_gt['all_images']
# 
#      print "Prepping for metric takes %f s." %(time.time() -t)
#      t = time.time()
#      score = compute_cider(cider_scorer, class_gen, new_gt_annotations, tfidf_dict)
#      print "Computing score takes %f s." %(time.time() - t)
#
#      print gen_cl, gt_cl, score
#      cider_matrix[gen_cl, gt_cl] = score 
#
#  pkl.dump(meteor_matrix, open('meteor_confusion/meteor_confusion_%s.p' %tag, 'w'))

if __name__ == '__main__':
   
  #  for annotation in gt_annotations
  parser = argparse.ArgumentParser()
  parser.add_argument("--tag", type=str, default='None')
  args = parser.parse_args()

  eval_class_meteor(args.tag)

