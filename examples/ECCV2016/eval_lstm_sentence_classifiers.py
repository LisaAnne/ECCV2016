import sys
sys.path.insert(0, '../../python/')
sys.path.insert(0, 'utils/')
import caffe
caffe.set_mode_gpu()
caffe.set_device(2)
from init import *
import re
import numpy as np
import argparse
import pdb
import copy
import scipy.io as sio
import pickle as pkl

SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')

def split_sentence(sentence):
  # break sentence into a list of words and punctuation
  sentence = [s.lower() for s in SENTENCE_SPLIT_REGEX.split(sentence.strip()) if len(s.strip()) > 0]
  if sentence[-1] != '.':
    return sentence
  return sentence[:-1]

def open_vocab(vocab_txt):
  vocab_list = open(vocab_txt).readlines()
  vocab_list = ['EOS'] + [v.strip() for v in vocab_list]
  vocab = {}
  for iv, v in enumerate(vocab_list): vocab[v] = iv 
  return vocab

def tokenize_text(sentence, vocabulary):
 sentence = split_sentence(sentence) 
 token_sent = []
 for w in sentence:
   try:
     token_sent.append(vocabulary[w])
   except:
     pass
     #token_sent.append(vocabulary['<unk>'])
 token_sent.append(vocabulary['EOS'])
 return token_sent

def compute_accuracy(pred, gt):
  num_correct = np.sum((np.array(pred) - np.array(gt)) == 0, dtype=int)   
  return float(num_correct)/len(pred)

def average_image_sents(image_ids, probs, gt):
  image_prob_dict = {}
  for ix, image_id in enumerate(image_ids):
    if image_id in image_prob_dict.keys():
      image_prob_dict[image_id] += probs[ix]
    else:
      image_prob_dict[image_id] = probs[ix]
  
  gt_av = []
  labels_av = []
  for key in image_prob_dict.keys():
    gt_av.append(int(key.split('.')[0]) - 1)
    labels_av.append(np.argmax(image_prob_dict[key]))

  print "Accuracy averaging over sentence: %f" %compute_accuracy(labels_av, gt_av)

def analyze_gen_net(model, model_weights, caps, save_classifications=False):
  net = caffe.Net(model, model_weights, caffe.TEST)

  vocab_txt = 'data/vocab.txt' 
  vocab = open_vocab(vocab_txt)
  vocab_size = len(vocab.keys())

  images = []
  captions = read_json(caps)
  token_captions = []
  for annotation in captions:
    token_sent = tokenize_text(annotation['caption'], vocab)
    class_label = int(annotation['image_id'].split('.')[0]) - 1
    token_captions.append((token_sent, class_label, annotation['image_id']))

  max_batch_size = 1000
  max_sentence = 20
  all_gt_labels = []
  all_probs_labels = []
  all_probs_probs = []
  for i in range(0,len(token_captions), max_batch_size):
    e = min(len(token_captions), i + max_batch_size)
    batch_size = min(e-i, max_batch_size)
    input_array = np.zeros((max_sentence, batch_size))     
    cont_array = np.ones((max_sentence, batch_size))
    cont_array[0,:] = 0
    gt_labels = np.zeros((batch_size,))

    count = 0
    sent_length = []
    for cap, l, image_id, in token_captions[i:e]:
      len_cap = min(len(cap), max_sentence)
      sent_length.append(len_cap - 1)
      input_array[:len_cap, count] = np.array(cap[:len_cap])
      gt_labels[count] = l 
      count += 1
      images.append(image_id)
 
    all_gt_labels.extend(gt_labels)
 
    net.blobs['input_sentence'].reshape(20, batch_size)
    net.blobs['cont_sentence'].reshape(20, batch_size)
    net.blobs['input_sentence'].data[...] = input_array
    net.blobs['cont_sentence'].data[...] = cont_array
    net.forward()

    probs = copy.deepcopy(net.blobs['probs'].data[...])

    probs_labels = []
    probs_probs = []
    for ix, sl in enumerate(sent_length):
      probs_labels.append(np.argmax(probs[sl,ix,:]))
      probs_probs.append(probs[sl,ix,:])
    all_probs_labels.extend(probs_labels)
    all_probs_probs.extend(probs_probs)

  print 'Accuracy: %f' %compute_accuracy(all_probs_labels, all_gt_labels)
  if save_classifications:
    save_classification_name =  'descriminator_output/' + caps.split('/')[-3] + '.p'
    output_classifications = {}
    output_classifications['images'] = images
    output_classifications['all_probs_labels'] = all_probs_labels
    output_classifications['all_gt_labels'] = all_gt_labels
    pkl.dump(output_classifications, open(save_classification_name, 'w')) 

  average_image_sents(zip(*token_captions)[2], all_probs_probs, all_gt_labels)

def analyze_net(model, model_weights):
  net = caffe.Net(model, model_weights, caffe.TEST)

  save_activation = False

  #arg_split = 'train_noCub'
  arg_split = 'val'
  val_caps = bird_anno_path_fg %(arg_split)
  vocab_txt = 'data/vocab.txt' 
  vocab = open_vocab(vocab_txt)
  vocab_size = len(vocab.keys())

  captions = read_json(val_caps)
  token_captions = []
  for annotation in captions['annotations']:
    token_sent = tokenize_text(annotation['caption'], vocab)
    class_label = int(annotation['image_id'].split('.')[0]) - 1
    token_captions.append((token_sent, class_label, annotation['image_id']))
  
  if save_activation:
    save_activation_lstm = np.zeros((len(token_captions), 1000))
    save_activation_ip = np.zeros((len(token_captions), 200))

  max_batch_size = 1000
  max_sentence = 20
  all_gt_labels = []
  all_probs_labels = []
  all_probs_probs = []
  all_images = []
  for i in range(0,len(token_captions), max_batch_size):
    e = min(len(token_captions), i + max_batch_size)
    batch_size = min(e-i, max_batch_size)
    input_array = np.zeros((max_sentence, batch_size))     
    cont_array = np.ones((max_sentence, batch_size))
    cont_array[0,:] = 0
    gt_labels = np.zeros((batch_size,))

    count = 0
    sent_length = []
    for cap, l, image_id, in token_captions[i:e]:
      all_images.append(image_id)
      len_cap = min(len(cap), max_sentence)
      sent_length.append(len_cap - 1)
      input_array[:len_cap, count] = np.array(cap[:len_cap])
      gt_labels[count] = l 
      count += 1
 
    all_gt_labels.extend(gt_labels)
 
    net.blobs['input_sentence'].reshape(20, batch_size)
    net.blobs['cont_sentence'].reshape(20, batch_size)
    net.blobs['input_sentence'].data[...] = input_array
    net.blobs['cont_sentence'].data[...] = cont_array
    net.forward()
    #pdb.set_trace()
   

    probs = copy.deepcopy(net.blobs['probs'].data[...])
 
    probs_labels = []
    probs_probs = []
    for ix, sl in enumerate(sent_length):
      probs_labels.append(np.argmax(probs[sl,ix,:]))
      probs_probs.append(probs[sl,ix,:])
      if save_activation:
        save_activation_lstm[i+ix,:] = copy.deepcopy(net.blobs['lstm'].data[sl, ix,:])
        save_activation_ip[i+ix,:] = copy.deepcopy(net.blobs['predict'].data[sl, ix,:])
    all_probs_labels.extend(probs_labels)
    all_probs_probs.extend(probs_probs)

  val_mat = np.zeros((len(all_probs_probs), 200))
  for i in range(len(all_probs_probs)):
    val_mat[i,:] = all_probs_probs[i]
  print 'Accuracy: %f' %compute_accuracy(all_probs_labels, all_gt_labels)

  average_image_sents(zip(*token_captions)[2], all_probs_probs, all_gt_labels)

  if save_activation:
    mat_file = '%s_%s.mat' %(model_weights.split('/')[-1].split('.caffemodel')[0], arg_split)
    sio.savemat(mat_file, {'files': all_images, 'activation_lstm': save_activation_lstm, 'activation_ip': save_activation_ip}) 
    print "Saved activations to %s." %(mat_file)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--model_name", type=str, default='prototxt/caption_classifier_embedDrop_75_lstmDrop_75_embedHidden_1000_lstmHidden_1000_deploy.prototxt')
  parser.add_argument("--model_weights",type=str, default='gve_models/caption_classifier_1006.caffemodel')
  parser.add_argument("--caps",type=str)
  parser.add_argument("--sentence_type", type=str, default='gt')
  args = parser.parse_args()

  if args.sentence_type == 'gt':
    analyze_net(args.model_name, args.model_weights)
  elif args.sentence_type == 'gen':
    analyze_gen_net(args.model_name, args.model_weights, args.caps)
  else:
    raise Exception("Not a valid sentence type.")



