import argparse
import sys
sys.path.append('utils/')
from eval import eval_generation
import pdb
from init import *
import random
import os
import copy
import numpy as np
import pickle as pkl

def shuffle_captions(args):
  #read gt captions
  image_root = eval_generation.determine_image_pattern(args.dataset_name, args.split_name)
  anno_path = eval_generation.determine_anno_path(args.dataset_name, args.split_name)
  #revise anno path
  gt_captions = read_json(anno_path)
  gt_captions_small = {}
  gt_captions_small['type'] = 'captions'
  gt_captions_small['images'] = []
  gt_captions_small['annotations'] = []
  gt_gen_captions = []
  
  im_to_captions = {}
  for a in gt_captions['annotations']:
    if a['image_id'] in im_to_captions.keys():
      im_to_captions[a['image_id']].append(a['caption'])
    else:
      im_to_captions[a['image_id']] = [a['caption']]
  
  count = 0
  for image_id in im_to_captions.keys():
    gt_caps = im_to_captions[image_id][1:]
    new_gt_a = [{'caption': gc, 'id':count+i, 'image_id': image_id} for i, gc in enumerate(gt_caps)] 
    count += len(gt_caps)
    new_gt_i = {'id': image_id, 'file_name': image_id}
    gt_captions_small['annotations'].extend(new_gt_a)
    gt_captions_small['images'].append(new_gt_i)

    new_val_a = {'image_id': image_id, 'caption':im_to_captions[image_id][0]}
    gt_gen_captions.append(new_val_a)
    
  save_json(gt_captions_small, 'tmp_gt_json.json')
  anno_path = os.getcwd() + '/tmp_gt_json.json'

  vocab_file = '%s/%s.txt' %(eval_generation.determine_vocab_folder(args.dataset_name, args.split_name), args.vocab)
  vocab = open_txt(vocab_file)

  sg = eval_generation.build_sequence_generator(anno_path, 100, image_root, 
                                  vocab = vocab, max_words=50)
  def compute_metrics(results):
    caption_experiment = eval_generation.CaptionExperiment(sg = sg)
    caption_experiment.score_generation(json_filename='tmp_json_out.json')

  def no_shuffle(val_generated):
    save_json(val_generated, 'tmp_json_out.json')
    compute_metrics('tmp_json_out.json')
    os.remove('tmp_json_out.json')

  def shuffle_all(gen_caps):
    all_caps = [g['caption'] for g in gen_caps]
    random.shuffle(all_caps)
    val_generated = []
    for count, key in enumerate(im_to_captions.keys()):
      val_generated.append({'image_id': key, 'caption': all_caps[count]})  
    save_json(val_generated, 'tmp_json_out.json')
    compute_metrics('tmp_json_out.json')
    os.remove('tmp_json_out.json')

  def shuffle_classes(gen_caps):
    val_classes = open_txt(bird_dataset_path + 'zero_shot_splits/valclasses.txt')   
    class_captions = {}
    for g in gen_caps:
      c = g['image_id'].split('/')[0]
      if c in class_captions.keys():
        class_captions[c].append(g['caption'])
      else:
        class_captions[c] = [copy.deepcopy(g['caption'])]
    for c in class_captions: random.shuffle(class_captions[c])
    count_classes = {}
    for c in class_captions: count_classes[c] = 0
  
    val_generated = []
    for g in gen_caps:
      c = g['image_id'].split('/')[0]
      class_caption = class_captions[c][count_classes[c]]
      count_classes[c] += 1
      val_generated.append({'image_id': g['image_id'], 'caption': class_caption})
    save_json(val_generated, 'tmp_json_out.json')
    compute_metrics('tmp_json_out.json')
    os.remove('tmp_json_out.json') 

  #shuffle gt captions
  print "Running shuffle experiments: No shuffle gt captions..."
  no_shuffle(gt_gen_captions)
  
  print "Running shuffle experiments: Randomly shuffle within class..."
  shuffle_classes(gt_gen_captions)

  print "Running shuffle experiments: Randomly shuffle gt captions..."
  shuffle_all(gt_gen_captions)

  gen_captions = read_json('generated_sentences/birds_from_scratch_zsSplit_freezeConv_iter_20000.generation_result.json')
  
  print "Running shuffle experiments: No shuffle lrcn captions..."
  no_shuffle(gen_captions)
  
  print "Running shuffle experiments: Randomly shuffle within class..."
  shuffle_classes(gen_captions)

  print "Running shuffle experiments: Randomly shuffle lrcn captions..."
  shuffle_all(gen_captions)

def nn_metrics():
  image_root = eval_generation.determine_image_pattern('birds_fg', '')
  vocab_file = '%s/%s.txt' %(eval_generation.determine_vocab_folder('birds_fg', ''), 'CUB_vocab_noUNK')
  vocab = open_txt(vocab_file)
  #gt json
  anno_path_train = eval_generation.determine_anno_path('birds_fg', 'test')
  sg = eval_generation.build_sequence_generator(anno_path_train, 100, image_root, 
                                  vocab = vocab, max_words=50)

  caption_experiment = eval_generation.CaptionExperiment(sg = sg)
  caption_experiment.score_generation(json_filename='generated_sentences/nearest_neighbor_baseline.json')
  

def repeat_captions(gen, gt):

  gen_json = read_json(gen)
  gen_sents = [c['caption'] for c in gen_json]

  gt_json = read_json(gt)
  gt_sents = [a['caption'] for a in gt_json['annotations']]  

  repeat = 0
  for c in gen_sents:
    if c in gt_sents:
      repeat += 1
  print 'Percent copied sentences is %f' %(float(repeat)/len(gen_sents))

def eval_cc_caffe_model(args):
  experiment = {'type': 'generation'}

  args.model_name = args.model_name.split(',')

  experiment = {'type': 'generation'}
  experiment['prev_word_restriction'] = args.prev_word

  pred = args.pred

  #set everything up
  captioner, sg, dataset = eval_generation.build_captioner(args.model_name, args.image_net, args.LM_net, args.dataset_name, args.split_name, args.vocab, args.precomputed_h5, args.gpu, experiment['prev_word_restriction']) 
 
  beam_size = 1 
  strategy = {'type': 'beam', 'beam_size': beam_size}
  strategy_name = 'beam%d' % strategy['beam_size']
  dataset_subdir = '%s_%s' % (args.dataset_name, args.split_name)
  dataset_cache_dir = '%s/%s/%s' % (cache_home, dataset_subdir, args.model_name[0])
  feature_cache_dir = '%s/%s/%s' % (cache_home, dataset_subdir, args.model_name[0])
  if pred:
    dataset_cache_dir = '%s/%s/%s_pred' % (cache_home, dataset_subdir, args.model_name[0])
    feature_cache_dir = '%s/%s/%s_pred' % (cache_home, dataset_subdir, args.model_name[0])
  cache_dir = '%s/%s' % (dataset_cache_dir, strategy_name)
  experimenter = eval_generation.CaptionExperiment(captioner, dataset, feature_cache_dir, cache_dir, sg)
  captioner.set_image_batch_size(min(100, len(dataset.keys())))

  #compute descriptors
  print 'Computing image descriptors'
  descriptor_labels = [df.split('/')[-2].split('.')[0] for df in experimenter.images]
  if pred:
    label_dict = pkl.load(open('/x/lisaanne/finegrained/bilinear_preds.p', 'r')) 
    descriptor_labels = [label_dict['/'.join(df.split('/')[-2:])] + 1 for df in experimenter.images]

  experimenter.compute_descriptors(des_file_idx=0, file_load=False)
  num_descriptors = experimenter.descriptors.shape[0]
  descriptor_files = experimenter.descriptor_filename

  size_input_feature = args.size_input_features
  concat_descriptors = np.zeros((num_descriptors, size_input_feature))

  num_descriptors = len(descriptor_labels)
  for i in range(num_descriptors):
    concat_descriptors[i,:1000] = experimenter.descriptors[i,:]

  if size_input_feature == 1001:
    for i in range(num_descriptors):
      concat_descriptors[i,-1] = float(descriptor_labels[i]) 
 
  if size_input_feature == 1200:
    for i in range(num_descriptors):
      binary_vec = np.zeros((200,))
      binary_vec[int(descriptor_labels[i])-1] = 1
      concat_descriptors[i,-200:] = binary_vec*args.label_scale 

  if size_input_feature == 2000:
    lookup_mat = pkl.load(open(args.lookup_mat, 'r'))
    for i in range(num_descriptors):
      lookup_index = int(descriptor_labels[i])-1
      concat_descriptors[i, -1000:] = lookup_mat[lookup_index,:] 
  experimenter.descriptors = concat_descriptors
 
  #generate descriptions
  max_batch_size = 1000
  num_images = len(experimenter.images)
  do_batches = (strategy['type'] == 'beam' and strategy['beam_size'] == 1) or \
      (strategy['type'] == 'sample' and
       ('temp' not in strategy or strategy['temp'] in (1, float('inf'))) and
       ('num' not in strategy or strategy['num'] == 1))
  batch_size = min(max_batch_size, num_images) if do_batches else 1

  all_captions = [None] * num_images
  image_index = 0
 
  all_captions, image_index = experimenter.generate_captions(strategy, do_batches, batch_size, image_index=image_index)

  experimenter.save_and_score_generation(all_captions)

  check_equiv = 37
  print descriptor_files[check_equiv]
  captions, caption_probs = experimenter.captioner.sample_captions([concat_descriptors[check_equiv]], temp=float('inf'), min_length = 2)
  print experimenter.captioner.sentence(captions[0])
  print experimenter.captioner.sentence(all_captions[check_equiv])

def eval_class_caffe_model(args):
  experiment = {'type': 'generation'}

  args.model_name = args.model_name.split(',')

  experiment = {'type': 'generation'}
  experiment['prev_word_restriction'] = args.prev_word
  pred = args.pred

  #set everything up
  captioner, sg, dataset = eval_generation.build_captioner(args.model_name, None, args.LM_net, args.dataset_name, args.split_name, args.vocab, None, args.gpu, experiment['prev_word_restriction']) 
 
  beam_size = 1 
  strategy = {'type': 'beam', 'beam_size': beam_size}
  strategy_name = 'beam%d' % strategy['beam_size']
  dataset_subdir = '%s_%s' % (args.dataset_name, args.split_name)
  dataset_cache_dir = '%s/%s/%s' % (cache_home, dataset_subdir, args.model_name[0])
  feature_cache_dir = '%s/%s/%s' % (cache_home, dataset_subdir, args.model_name[0])
  if pred:
    dataset_cache_dir = '%s/%s/%s_pred' % (cache_home, dataset_subdir, args.model_name[0])
    feature_cache_dir = '%s/%s/%s_pred' % (cache_home, dataset_subdir, args.model_name[0])
  cache_dir = '%s/%s' % (dataset_cache_dir, strategy_name)
  experimenter = eval_generation.CaptionExperiment(captioner, dataset, feature_cache_dir, cache_dir, sg)

  experimenter.descriptor_filename = experimenter.images
  num_descriptors = len(experimenter.descriptor_filename)
  descriptor_files = experimenter.descriptor_filename
  #add class condition
  descriptor_labels = [df.split('/')[-2].split('.')[0] for df in descriptor_files]
  if pred:
    label_dict = pkl.load(open('/yy2/lisaanne/fine_grained/bilinear_features/finegrained/bilinear_preds.p', 'r')) 
    descriptor_labels = [label_dict['/'.join(df.split('/')[-2:])] + 1 for df in experimenter.images]

  size_input_feature = args.size_input_features 
  concat_descriptors = np.zeros((num_descriptors, size_input_feature))
  if size_input_feature == 200:
    for i in range(num_descriptors):
      binary_vec = np.zeros((200,))
      binary_vec[int(descriptor_labels[i])-1] = 1
      concat_descriptors[i,] = binary_vec 

  if size_input_feature == 1000:
    lookup_mat = pkl.load(open(args.lookup_mat, 'r'))
    for i in range(num_descriptors):
      lookup_index = int(descriptor_labels[i])-1
      concat_descriptors[i, :] = lookup_mat[lookup_index,:]
 
  experimenter.descriptors = concat_descriptors
 
  #generate descriptions
  max_batch_size = 1000
  num_images = len(experimenter.images)
  do_batches = (strategy['type'] == 'beam' and strategy['beam_size'] == 1) or \
      (strategy['type'] == 'sample' and
       ('temp' not in strategy or strategy['temp'] in (1, float('inf'))) and
       ('num' not in strategy or strategy['num'] == 1))
  batch_size = min(max_batch_size, num_images) if do_batches else 1

  all_captions = [None] * num_images
  image_index = 0
 
  all_captions, image_index = experimenter.generate_captions(strategy, do_batches, batch_size, image_index=image_index)
  experimenter.save_and_score_generation(all_captions)

def eval_caffe_model(args):
  experiment = {'type': 'generation'}

  args.model_name = args.model_name.split(',')

  experiment = {'type': 'generation'}
  experiment['prev_word_restriction'] = args.prev_word

  eval_generation.main(model_name=args.model_name, image_net=args.image_net, LM_net=args.LM_net, dataset_name=args.dataset_name, split_name=args.split_name, vocab=args.vocab, precomputed_h5=args.precomputed_h5, experiment=experiment, gpu=args.gpu)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--model_name",type=str)
  parser.add_argument("--image_net",type=str)
  parser.add_argument("--LM_net",type=str)
  parser.add_argument("--dataset_name",type=str,default='coco')
  parser.add_argument("--split_name",type=str,default='val_val')
  parser.add_argument("--vocab",type=str,default='vocabulary')
  parser.add_argument("--precomputed_feats",type=str,default=None)
  parser.add_argument("--precomputed_h5",type=str,default=None)
  parser.add_argument("--gpu",type=int,default=0)
  parser.add_argument("--size_input_features", type=int, default=1000)
  parser.add_argument("--experiment_type", type=str, default='eval_caffe_model')
  parser.add_argument("--label_scale", type=int, default=1)
  parser.add_argument("--lookup_mat", type=str, default='utils_fineGrained/class_embedding/class_lrcn.p')
  parser.add_argument("--gen_caps", type=str)
  parser.add_argument("--gt_caps", type=str)
  parser.add_argument('--prev_word_restriction', dest='prev_word', action='store_true')
  parser.set_defaults(prev_word=False)
  parser.add_argument('--pred', dest='pred', action='store_true')
  parser.set_defaults(pred=False)

  args = parser.parse_args()

  if args.experiment_type == 'eval_caffe_model':
    eval_caffe_model(args)
  elif args.experiment_type == 'eval_cc_caffe_model':
    eval_cc_caffe_model(args)
  elif args.experiment_type == 'eval_class_caffe_model':
    eval_class_caffe_model(args)
  elif args.experiment_type == 'shuffle_captions':
    shuffle_captions(args) 
  elif args.experiment_type == 'repeat_captions':
    repeat_captions(args.gen_caps, args.gt_caps)
  elif args.experiment_type == 'nn_metrics':
    nn_metrics()
  else:
    raise Exception("Did not select valid experiment type.") 



