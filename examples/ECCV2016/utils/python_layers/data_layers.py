#!/usr/bin/env python

import pdb
pycaffe_path = '/home/lisaanne/caffe-LSTM/python/'
import sys
sys.path.append('../../python/')
sys.path.append('utils/')
import caffe
import io
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import time
import glob
import pickle as pkl
import random
import h5py
from multiprocessing import Pool
from threading import Thread
import skimage.io
import copy
import json
import hickle as hkl
import time
import re
import math
from init import *

UNK_IDENTIFIER = '<unk>'
SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')

def read_json(t_file):
  j_file = open(t_file).read()
  return json.loads(j_file)

def split_sentence(sentence):
  # break sentence into a list of words and punctuation
  sentence = [s.lower() for s in SENTENCE_SPLIT_REGEX.split(sentence.strip()) if len(s.strip()) > 0]
  if sentence[-1] != '.':
    return sentence
  return sentence[:-1]
 
def tokenize_text(sentence, vocabulary, leave_out_unks=False):
 sentence = split_sentence(sentence) 
 token_sent = []
 for w in sentence:
   try:
     token_sent.append(vocabulary[w])
   except:
     if not leave_out_unks:
       try:
         token_sent.append(vocabulary['<unk>'])
       except:
         pass
     else:
       pass
 if not leave_out_unks:
   token_sent.append(vocabulary['EOS'])
 return token_sent

def open_vocab(vocab_txt):
  vocab_list = open(vocab_txt).readlines()
  vocab_list = ['EOS'] + [v.strip() for v in vocab_list]
  vocab = {}
  for iv, v in enumerate(vocab_list): vocab[v] = iv 
  return vocab

def cub_labels(annotation):
  image_id = annotation['image_id']
  return int(image_id.split('.')[0]) - 1  

def cub_labels_gen(annotation, labels):
  return labels[annotation['image_id']]

def textPreprocessor(params):
  #input: 
  #     params['caption_json']: text json which contains text and a path to an image if the text is grounded in an image
  #     params['vocabulary']: vocabulary txt to use
  #     params['label_extract']: dataset to direct label extraction or None
  #output:
  #     processed_text: tokenized text with corresponding image path (if they exist)

  #make vocabulary dict
  vocab = open_vocab(params['vocabulary'])
  json_text = read_json(params['caption_json'])
  processed_text = {}
  if 'label_extract' not in params.keys():  params['label_extract'] = None 
  if 'length_label' not in params.keys(): params['length_label'] = None
  if params['label_extract'] == 'CUB_gen':
    #read in predicted labels
    labels = pkl.load('/x/lisaanne/finegrained/CUB_label_dict.p') 
    cub_labels_gen_prep = lambda x: cub_labels_gen(x, labels)
  else:
    cub_labels_gen_prep = cub_labels
 
  label_function = {'CUB': cub_labels, 'CUB_gen': cub_labels_gen_prep}




  t = time.time()
  for annotation in json_text['annotations']:
    processed_text[annotation['id']] = {}
    processed_text[annotation['id']]['text'] = tokenize_text(annotation['caption'], vocab)
    if 'image_id' in annotation.keys():
      processed_text[annotation['id']]['image'] = annotation['image_id']
    if params['label_extract']:
      processed_text[annotation['id']]['label'] = label_function[params['label_extract']](annotation) 
    if params['length_label']:
      processed_text[annotation['id']]['length'] = len(annotation['caption'])
  print "Setting up text dict: ", time.time()-t
 
  return processed_text 

class extractData(object):

  def increment(self): 
  #uses iteration, batch_size, data_list, and num_data to extract next batch identifiers
    next_batch = [None]*self.batch_size
    if self.iteration + self.batch_size >= self.num_data:
      next_batch[:self.num_data-self.iteration] = self.data_list[self.iteration:]
      next_batch[self.num_data-self.iteration:] = self.data_list[:self.batch_size -(self.num_data-self.iteration)]
      random.shuffle(self.data_list)
      self.iteration = self.num_data - self.iteration
    else:
      next_batch = self.data_list[self.iteration:self.iteration+self.batch_size]
      self.iteration += self.batch_size
    assert self.iteration > -1
    assert len(next_batch) == self.batch_size 
    return next_batch
 
  def advanceBatch(self):
    next_batch = self.increment()
    self.get_data(next_batch)

class extractFeatureText(extractData):

  def __init__(self, dataset, params, result):
    self.extractType = 'text'
    self.data_list = dataset.keys() 
    self.num_data = len(self.data_list)
    print 'For extractor extractText, length of data is: ', self.num_data
    self.dataset = dataset
    self.iteration = 0
    self.image_dim = params['image_dim']
    self.batch_size = params['batch_size']
    self.stream_size = params['stream_size']
    if not 'feature_size' in params.keys():
      params['feature_size'] = 8192
    assert 'features' in params.keys()
    self.feature_size = params['feature_size']

    self.features = params['features']

    #preperation to output top
    self.text_data_key = params['text_data_key']
    self.text_label_key = params['text_label_key']
    self.marker_key = params['text_marker_key']
    self.image_data_key = params['image_data_key']
    self.top_keys = [self.text_data_key, self.text_label_key, self.marker_key, self.image_data_key]
    self.batch_size = params['batch_size']
    self.stream_size = params['stream_size']
    self.top_shapes = [(self.stream_size, self.batch_size), (self.stream_size, self.batch_size), (self.stream_size, self.batch_size), (self.batch_size, self.feature_size)]
    self.result = result 
 
  def get_data(self, next_batch):
    batch_images = [self.dataset[nb]['image'] for nb in next_batch]
    next_batch_input_sentences = np.zeros((self.stream_size, self.batch_size))
    next_batch_target_sentences = np.ones((self.stream_size, self.batch_size))*-1
    next_batch_markers = np.ones((self.stream_size, self.batch_size))
    next_batch_image_data = np.ones((self.batch_size, self.feature_size))
    next_batch_markers[0,:] = 0
    for ni, nb in enumerate(next_batch):
      ns = self.dataset[nb]['text']
      num_words = len(ns)
      ns_input = ns[:min(num_words, self.stream_size-1)]
      ns_target = ns[:min(num_words, self.stream_size)]
      next_batch_input_sentences[1:min(num_words+1, self.stream_size), ni] = ns_input 
      next_batch_target_sentences[:min(num_words, self.stream_size), ni] = ns_target
      for ni, nb in enumerate(batch_images):
        next_batch_image_data[ni,...] = self.features[nb] 

    self.result[self.text_data_key] = next_batch_input_sentences
    self.result[self.text_label_key] = next_batch_target_sentences
    self.result[self.marker_key] = next_batch_markers
    self.result[self.image_data_key] = next_batch_image_data

class extractImageText(extractData):

  def __init__(self, dataset, params, result):
    self.extractType = 'text'
    self.data_list = dataset.keys() 
    self.num_data = len(self.data_list)
    print 'For extractor extractText, length of data is: ', self.num_data
    self.dataset = dataset
    self.iteration = 0
    self.image_dim = params['image_dim']
    self.batch_size = params['batch_size']
    self.stream_size = params['stream_size']
    self.base_path = params['base_image_path']

    #prep to process image
    image_data_shape = (self.batch_size, 3, self.image_dim, self.image_dim)
    self.transformer = define_transformer(params['image_data_key'], image_data_shape, self.image_dim) 
    self.imageProcessor = imageProcessor(self.transformer, self.image_dim, params['image_data_key']) 

    #preperation to output top
    self.text_data_key = params['text_data_key']
    self.text_label_key = params['text_label_key']
    self.marker_key = params['text_marker_key']
    self.image_data_key = params['image_data_key']
    self.top_keys = [self.text_data_key, self.text_label_key, self.marker_key, self.image_data_key]
    self.batch_size = params['batch_size']
    self.stream_size = params['stream_size']
    self.top_shapes = [(self.stream_size, self.batch_size), (self.stream_size, self.batch_size), (self.stream_size, self.batch_size), image_data_shape]
    self.result = result 
    
    self.pool_size = 4
    self.pool = Pool(processes=self.pool_size)
 
  def get_data(self, next_batch):
    batch_images = ['/'.join([self.base_path, self.dataset[nb]['image']]) for nb in next_batch]
    next_batch_input_sentences = np.zeros((self.stream_size, self.batch_size))
    next_batch_target_sentences = np.ones((self.stream_size, self.batch_size))*-1
    next_batch_markers = np.ones((self.stream_size, self.batch_size))
    next_batch_image_data = np.ones((self.batch_size, 3, self.image_dim, self.image_dim))
    next_batch_markers[0,:] = 0
    for ni, nb in enumerate(next_batch):
      ns = self.dataset[nb]['text']
      num_words = len(ns)
      ns_input = ns[:min(num_words, self.stream_size-1)]
      ns_target = ns[:min(num_words, self.stream_size)]
      next_batch_input_sentences[1:min(num_words+1, self.stream_size), ni] = ns_input 
      next_batch_target_sentences[:min(num_words, self.stream_size), ni] = ns_target
    if self.pool_size > 1:
      next_batch_images_list = self.pool.map(self.imageProcessor, batch_images)
      for ni in range(len(next_batch)):
        next_batch_image_data[ni,...] = next_batch_images_list[ni]
    else:
      for ni, nb in enumerate(batch_images):
        next_batch_image_data[ni,...] = self.imageProcessor(nb) 

    self.result[self.text_data_key] = next_batch_input_sentences
    self.result[self.text_label_key] = next_batch_target_sentences
    self.result[self.marker_key] = next_batch_markers
    self.result[self.image_data_key] = next_batch_image_data

class extractText(extractData):

  def __init__(self, dataset, params, result):
    self.extractType = 'text'
    self.data_list = dataset.keys() 
    self.num_data = len(self.data_list)
    print 'For extractor extractText, length of data is: ', self.num_data
    self.dataset = dataset
    self.iteration = 0
    self.batch_size = params['batch_size']
    self.stream_size = params['stream_size']

    #preperation to output top
    self.text_data_key = params['text_data_key']
    self.text_label_key = params['text_label_key']
    self.marker_key = params['text_marker_key']
    self.top_keys = [self.text_data_key, self.text_label_key, self.marker_key]
    self.batch_size = params['batch_size']
    self.stream_size = params['stream_size']
    self.top_shapes = [(self.stream_size, self.batch_size), (self.stream_size, self.batch_size), (self.stream_size, self.batch_size)]
    self.result = result 
    
  def get_data(self, next_batch):
    next_batch_input_sentences = np.zeros((self.stream_size, self.batch_size))
    next_batch_target_sentences = np.ones((self.stream_size, self.batch_size))*-1
    next_batch_markers = np.ones((self.stream_size, self.batch_size))
    next_batch_markers[0,:] = 0
    for ni, nb in enumerate(next_batch):
      ns = self.dataset[nb]['text']
      num_words = len(ns)
      ns_input = ns[:min(num_words, self.stream_size-1)]
      ns_target = ns[:min(num_words, self.stream_size)]
      next_batch_input_sentences[1:min(num_words+1, self.stream_size), ni] = ns_input 
      next_batch_target_sentences[:min(num_words, self.stream_size), ni] = ns_target

    self.result[self.text_data_key] = next_batch_input_sentences
    self.result[self.text_label_key] = next_batch_target_sentences 
    self.result[self.marker_key] = next_batch_markers

class extractLabel(extractData):

  def __init__(self, dataset, params, result):
    self.data_list = dataset.keys() 
    self.num_data = len(self.data_list)
    print 'For extractor extractText, length of data is: ', self.num_data
    if 'label_format' not in params.keys():
      params['label_format'] = 'number'
    self.label_format = params['label_format'] #options: number, onehot, vector
					#number is just a number label
					#onehot is a onehot binary vector
					#vector requires a look up which maps number to continuous valued vector
    if self.label_format == 'vector':
      assert 'vector_file' in params.keys()
      lookup_file = params['vector_file'] #should be pkl file with a single matrix in it that is Label X EmbeddingD
      self.lookup = pkl.load(open(lookup_file, 'r'))
    if self.label_format == 'onehot':
      assert 'size_onehot' in params.keys()
    #determine label_size
    if self.label_format == 'number':
      self.label_size = 1
    elif self.label_format == 'onehot':
      self.label_size = params['size_onehot']
    elif self.label_format == 'vector':
      self.label_size = self.lookup.shape[1]

    self.dataset = dataset
    self.iteration = 0
    self.batch_size = params['batch_size']
    if 'label_stream_size' in params.keys():
      self.stream_size = params['label_stream_size']
    else:
      self.stream_size = 1
    self.supervision = params['sentence_supervision'] #'all' or 'last'

    #need to define 

    #preperation to output top
    self.label_key = params['data_label']
    self.top_keys = [self.label_key]
    if self.stream_size == 1:
      self.top_shapes = [(self.batch_size,self.label_size)]
    else:
      if self.label_size > 1:
        self.top_shapes = [(self.stream_size, self.batch_size, self.label_size)]
      else:
        self.top_shapes = [(self.stream_size, self.batch_size)]
    self.result = result 
   
  def number_label(self, nb):
    return nb 

  def onehot_label(self, nb):
    l = np.zeros((self.label_size,))
    l[nb] = 1
    return l

  def vector_label(self, nb):
    return self.lookup[nb,:]
 
  def get_data(self, next_batch):
    label_transform_dict = {'number': self.number_label, 'onehot': self.onehot_label, 'vector': self.vector_label}
    label_transform = label_transform_dict[self.label_format]
    next_batch_labels = np.ones((self.top_shapes[0]))*-1
    for ni, nb in enumerate(next_batch):
      nl = label_transform(self.dataset[nb]['label']) 
      if (self.supervision == 'all') & (self.stream_size > 1):
        next_batch_labels[:, ni] = nl
      if (self.supervision == 'last') & (self.stream_size > 1):
        #requires that 'target sentence' computed somewhere
        if len(np.where(self.result['target_sentence'][:,ni] == 0)[0] > 0):
          next_batch_labels[np.where(self.result['target_sentence'][:,ni] == 0)[0][0], ni] = nl
        else:
          next_batch_labels[-1, ni] = nl
          
      if (self.supervision == 'all') & (self.stream_size == 1):
        next_batch_labels[ni,:] = nl
      if (self.supervision == 'last') & (self.stream_size == 1):
        raise Exception("Cannot have 'last' supervision type if stream size is 1")

    self.result[self.label_key] = next_batch_labels

class extractLength(extractData):

  def __init__(self, dataset, params, result):
    self.data_list = dataset.keys() 
    self.num_data = len(self.data_list)
    self.dataset = dataset
    self.iteration = 0
    self.batch_size = params['batch_size']

    #preperation to output top
    self.length_key = params['length_label']
    self.top_keys = [self.length_key]
    self.top_shapes = [(self.batch_size,)]
    self.result = result 
   
  def get_data(self, next_batch):
    next_batch_length = np.ones((self.top_shapes[0]))*-1
    for ni, nb in enumerate(next_batch):
      nl = self.dataset[nb]['length'] 
      next_batch_length[ni] = nl

    self.result[self.length_key] = next_batch_length

class extractMulti(extractData):
  #extracts multiple bits of data from the same datasets (e.g., used to image description and image label)

  def __init__(self, dataset, params, result):
    #just need to set up parameters for "increment"
    self.extractors = params['extractors']
    self.batch_size = params['batch_size']
    self.data_list = dataset.keys() 
    self.num_data = len(self.data_list)
    self.dataset = dataset
    self.iteration = 0
    self.batch_size = params['batch_size']
  
    self.top_keys = []
    self.top_shapes = []
    for e in self.extractors:
      self.top_keys.extend(e.top_keys)
      self.top_shapes.extend(e.top_shapes)

  def get_data(self, next_batch):
    t = time.time()
    for e in self.extractors:
      e.get_data(next_batch)

class batchAdvancer(object):
  
  def __init__(self, extractors):
    self.extractors = extractors

  def __call__(self):
    #The batch advancer just calls each extractor
    for e in self.extractors:
      e.advanceBatch() 

class python_data_layer(caffe.Layer):
  
  def setup(self, bottom, top):
    random.seed(10)
  
    self.params = eval(self.param_str)
    params = self.params

    #set up prefetching
    self.thread_result = {}
    self.thread = None

    self.setup_extractors()
 
    self.batch_advancer = batchAdvancer(self.data_extractors) 
    self.top_names = []
    self.top_shapes = []
    for de in self.data_extractors:
      self.top_names.extend(de.top_keys)
      self.top_shapes.extend(de.top_shapes)
 
    self.dispatch_worker()

    if 'top_names' in params.keys():
      #check top names equal to each other...
      if not (set(params['top_names']) == set(self.top_names)):
        raise Exception("Input 'top names' not the same as determined top names.")
      else:
        self.top_names == params['top_names']

    print self.top_names
    print 'Outputs:', self.top_names
    if len(top) != len(self.top_names):
      raise Exception('Incorrect number of outputs (expected %d, got %d)' %
                      (len(self.top_names), len(top)))
    self.join_worker()
    #for top_index, name in enumerate(self.top_names.keys()):

    for top_index, name in enumerate(self.top_names):
      shape = self.top_shapes[top_index] 
      print 'Top name %s has shape %s.' %(name, shape)
      top[top_index].reshape(*shape)

  def reshape(self, bottom, top):
    pass

  def forward(self, bottom, top):
  
    if self.thread is not None:
      self.join_worker() 

    for top_index, name in zip(range(len(top)), self.top_names):
      top[top_index].data[...] = self.thread_result[name] 

    self.dispatch_worker()
      
  def dispatch_worker(self):
    assert self.thread is None
    self.thread = Thread(target=self.batch_advancer)
    self.thread.start()

  def join_worker(self):
    assert self.thread is not None
    self.thread.join()
    self.thread = None

  def backward(self, top, propagate_down, bottom):
    pass

class CaptionToLabel(python_data_layer):
 
  #Extracts data to train sentence classifier
 
  def setup_extractors(self):

    params = self.params

    #check that all parameters are included and set default params
    assert 'caption_json' in self.params.keys()
    assert 'vocabulary' in self.params.keys()

    if 'text_data_key' not in params.keys(): params['text_data_key'] = 'input_sentence'
    if 'text_label_key' not in params.keys(): params['text_label_key'] = 'target_sentence'
    if 'text_marker_key' not in params.keys(): params['text_marker_key'] = 'cont_sentence'
    if 'data_label' not in params.keys(): params['data_label'] = 'data_label'
    if 'batch_size' not in params.keys(): params['batch_size'] = 100 
    if 'stream_size' not in params.keys(): params['stream_size'] = 20 
    if 'sentence_supervision' not in params.keys(): params['sentence_supervision'] = 'all' #'all' vs. 'last'   
    if 'label_extract' not in params.keys(): params['label_extract'] = 'CUB'

  
    data = textPreprocessor(params)
    text_extractor = extractText(data, params, self.thread_result)

    if 'label_stream_size' not in params.keys():
      params['label_stream_size'] = params['stream_size']
    if 'data_label_feat' in params.keys():
      data_label = params['data_label']
      params['data_label'] = params['data_label_feat']
    label_extractor = extractLabel(data, params, self.thread_result)

    params['extractors'] = [text_extractor, label_extractor]
    multi_extractor = extractMulti(data, params, self.thread_result)
    self.data_extractors = [multi_extractor]

class extractGVEFeatures(python_data_layer):

  #Extract features for generating visual explanations
  #  input sentence: input words for each time step
  #  target sentence: target words for eath time step
  #  image features: iamge features
  #  data_label: class label
  #  data_label_feat: class label feature 
 
  def setup_extractors(self):

    params = self.params

    #check that all parameters are included and set default params
    assert 'caption_json' in self.params.keys()
    assert 'vocabulary' in self.params.keys()

    if 'text_data_key' not in params.keys(): params['text_data_key'] = 'input_sentence'
    if 'text_label_key' not in params.keys(): params['text_label_key'] = 'target_sentence'
    if 'text_marker_key' not in params.keys(): params['text_marker_key'] = 'cont_sentence'
    if 'data_label' not in params.keys(): params['data_label'] = 'data_label'
    if 'data_label_feat' not in params.keys(): params['data_label_feat'] = 'data_label_feat'
    if 'image_data_key' not in params.keys(): params['image_data_key'] = 'image_data'
    if 'batch_size' not in params.keys(): params['batch_size'] = 100 
    if 'stream_size' not in params.keys(): params['stream_size'] = 20 
    if 'image_dim' not in params.keys(): params['image_dim'] = 227 
    if 'sentence_supervision' not in params.keys(): params['sentence_supervision'] = 'all' #'all' vs. 'last'   
    if 'label_extract' not in params.keys(): params['label_extract'] = 'CUB'
    #assert 'vector_file' in params.keys()

 
    data = textPreprocessor(params)
    features = pkl.load(open(cub_features, 'r'))
    params['features'] = features

    imageText_extractor = extractFeatureText(data, params, self.thread_result)
    params['stream_size'] = 1

    #extract number label for loss layer
    params['label_format'] = 'number'
    label_extractor_number = extractLabel(data, params, self.thread_result)
    params['data_label'] = params['data_label_feat']

    params['label_format'] = 'vector'

    label_extractor_vector = extractLabel(data, params, self.thread_result)
    params['extractors'] = [imageText_extractor, label_extractor_number, label_extractor_vector]
    multi_extractor = extractMulti(data, params, self.thread_result)

    self.data_extractors = [multi_extractor]
  
