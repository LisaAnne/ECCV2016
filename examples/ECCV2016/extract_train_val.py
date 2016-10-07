import pdb
import sys
sys.path.insert(0, '../../python/')
from eval import eval_generation
import caffe
caffe.set_mode_gpu()
caffe.set_device(0)
import numpy as np
sys.path.append('utils/')
from init import *
import argparse
import scipy.io as sio
import copy
import pickle as pkl

image_input = True 

def extract_train_val(args):

  #Initialize captions
  experiment = {'type': 'generation'}
  experiment['prev_word_restriction'] = args.prev_word
  strategy_name = 'gt'
  dataset_subdir = '%s_%s' % (args.dataset_name, args.split_name)
  dataset_cache_dir = '%s/%s/%s' % (cache_home, dataset_subdir, args.model_name[0])
  feature_cache_dir = '%s/%s/%s' % (cache_home, dataset_subdir, args.model_name[0])
  cache_dir = '%s/%s' % (dataset_cache_dir, strategy_name)
  captioner, sg, dataset = eval_generation.build_captioner([args.model_name], args.image_net, args.LM_net, args.dataset_name, args.split_name, args.vocab, args.precomputed_h5, args.gpu, experiment['prev_word_restriction']) 
  save_activation = 'lstm2'
  experimenter = eval_generation.CaptionExperiment(captioner, dataset, feature_cache_dir, cache_dir, sg)
  
  experimenter.descriptor_filename = experimenter.images
  num_descriptors = len(experimenter.descriptor_filename)
  experimenter.compute_descriptors(0)
  descriptor_files = experimenter.descriptor_filename
  #add class condition
  descriptor_labels = [df.split('/')[-2].split('.')[0] for df in descriptor_files]

#  size_input_feature = 200
#  concat_descriptors = np.zeros((num_descriptors, size_input_feature))
#  for i in range(num_descriptors):
#    binary_vec = np.zeros((200,))
#    binary_vec[int(descriptor_labels[i])-1] = 1
#    concat_descriptors[i,] = binary_vec 
# 
#  experimenter.descriptors = concat_descriptors

  descriptor_dict = {}
  for name, des in zip(experimenter.descriptor_filename, experimenter.descriptors):
    descriptor_dict[name] = des
 
  #generate sentences
 
  cont = np.ones((20,1000))
  cont[0,:] = 0
  input_sent = np.zeros((20,1000))

  save_activation_mat = np.zeros((20, len(experimenter.captions), 1000))

  net = captioner.lstm_nets[0]

  max_batch_size = 1000
  im_list = []
  for i in range(0, len(experimenter.captions), max_batch_size):
    print i, len(experimenter.captions)
    batch_size = min(max_batch_size, len(experimenter.captions)-i)
    image_features = np.zeros((batch_size, args.size_input_feature))
    cont_in = cont[:,:batch_size] 
    sent_in = input_sent[:,:batch_size]
    for idx, caption in enumerate(experimenter.captions[i:i+batch_size]):
      c = caption['caption']
      im = caption['source_image']
      sent_in[0:min(20, len(c)), idx] = c[:min(20, len(c))] 
      image_features[idx,:] = descriptor_dict[im]     
      im_list.append(im)

    net.blobs['cont_sentence'].reshape(20, batch_size)
    net.blobs['input_sentence'].reshape(20, batch_size)

    net.blobs['cont_sentence'].data[...] = cont_in
    net.blobs['input_sentence'].data[...] = sent_in

    if image_input:
      net.blobs['image_features'].reshape(batch_size, args.size_input_feature)
      net.blobs['image_features'].data[...] = image_features
 
    net.forward()

    save_activation_mat[:, i:i+batch_size, :] = copy.deepcopy(net.blobs[save_activation].data)
  average_weights = np.zeros((save_activation_mat.shape[1], save_activation_mat.shape[2]))
  for ix, caption in enumerate(experimenter.captions):
    len_cap = min(20,len(caption['caption']))
    average_weights[ix,:] = np.mean(save_activation_mat[:len_cap,ix,:], axis=0)

#  mat_file = '%s_%s_gt_0930.mat' %(args.model_name.split('/')[-1], args.split_name)
#  sio.savemat(mat_file, {'files': im_list, 'average_weights': average_weights})
#  print "Saved mat file to %s." %mat_file

  class_weights = np.zeros((200, 1000))
  class_count = np.zeros((200,))
  for i, f in enumerate(im_list):
    c = int(f.split('/')[-2].split('.')[0]) - 1
    class_weights[c,:] += average_weights[i,:]
    class_count[c] += 1 

  for i in range(200):
    class_weights[i,:] /= class_count[i]

  save_name = 'data/%s_%s_gt_0930.p' %(args.model_name.split('/')[-1], args.split_name)
  pkl.dump(class_weights, open(save_name, 'w'))
  print "Wrote file to: %s" %save_name 

if __name__== '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--model_name",type=str)
  parser.add_argument("--image_net",type=str,default=None)
  parser.add_argument("--LM_net",type=str)
  parser.add_argument("--dataset_name",type=str,default='coco')
  parser.add_argument("--split_name",type=str,default='val_val')
  parser.add_argument("--vocab",type=str,default='vocabulary')
  parser.add_argument("--precomputed_feats",type=str,default=None)
  parser.add_argument("--precomputed_h5",type=str,default=None)
  parser.add_argument("--gpu",type=int,default=0)
  parser.add_argument("--experiment_type", type=str, default='eval_caffe_model')
  parser.add_argument("--size_input_feature", type=int, default=1200)
  parser.add_argument('--prev_word_restriction', dest='prev_word', action='store_true')
  parser.set_defaults(prev_word=False)

  args = parser.parse_args()

  #generate_hidden_class(args)
  extract_train_val(args)

