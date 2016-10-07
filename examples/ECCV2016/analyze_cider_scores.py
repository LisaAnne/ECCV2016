''' Code for class relevance metrics '''

import pickle as pkl
import numpy as np
import pdb
import copy
import os

tags = {'description': 'description', 
        'definition': 'definition',
        'explanation-label': 'explanation-label', 
        'explanation-dis': 'explanation-dis', 
        'explanation': 'explanation'}

cider_template = 'cider_scores/cider_score_dict_%s.p'
  
def compute_class_relevance_metrics():

  restrict_set = True  #only evaluate images which were correctly classified.  I did this because it is unclear if generated text for incorrect predictions should resemble text from the ground truth class or predicted class 
  good_ims = {}
  gen_labels = pkl.load(open('data/bilinear_preds.p'))
  count_good_ims = 0
  for im in gen_labels.keys():
    gt_label = int(im.split('/')[-2].split('.')[0]) - 1
    if gen_labels[im] == gt_label:
      good_ims[im] = True
      count_good_ims += 1
    else: 
      good_ims[im] = False

  for tag in tags.keys():
    print tag
    cider_dict = pkl.load(open(cider_template %tags[tag], 'r')) 
    ims = sorted(cider_dict.keys())    
    rank_mat_dict = {}
    for ix, im in enumerate(ims):
      orig_im = '_'.join(im.split('_')[:-1])
      if good_ims[orig_im]:
        if orig_im not in rank_mat_dict.keys():
          rank_mat_dict[orig_im] = np.zeros((200,))     
        cl_comp = int(im.split('_')[-1])
        rank_mat_dict[orig_im][cl_comp] = cider_dict[im]

    mean_rank = 0
    cider_similarity = 0
    for im in rank_mat_dict.keys():
      cl = int(im.split('.')[0]) - 1
      cider_max = np.argsort(rank_mat_dict[im])[::-1]
      mean_rank += np.where(cider_max == cl)[0][0] + 1 
      cider_similarity += rank_mat_dict[im][cl]

    print 'Mean cider similarity %s: %f' %(tag, cider_similarity/float(len(rank_mat_dict.keys())))
    print 'Mean rank %s: %f' %(tag, mean_rank/float(len(rank_mat_dict.keys())))

compute_class_relevance_metrics()
