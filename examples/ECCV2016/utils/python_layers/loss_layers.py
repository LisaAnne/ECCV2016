import sys
sys.path.insert(0, '../../python/')
import caffe
import numpy as np
import pdb
import time
from init import *

def create_vocab_dict(vocab):
  vocab_dict = {}
  for ix, v in enumerate(vocab):
    vocab_dict[v] = ix
  return vocab_dict 

def read_vocab(vocab_txt):
  vocab = open(vocab_txt).readlines()
  vocab = [v.strip() for v in vocab]
  return ['EOS'] + vocab 

def read_txt(txt_file):
  txt = open(txt_file).readlines()
  return [t.strip() for t in txt]

class avoidWordsLayer(caffe.Layer):

  def setup(self, bottom, top):
    #params: vocabulary, which words we should avoid
    params = eval(self.param_str)
    sampled_words = bottom[0].data

    self.vocab = read_vocab(params['vocab'])
    self.vocab_dict = create_vocab_dict(self.vocab)
    self.avoid_words = params['avoid_words']
    self.size_vocab = len(self.vocab)
    self.T = sampled_words.shape[0]   
    self.N = sampled_words.shape[1]  
    if 'baseline' in params.keys():
      self.baseline = params['baseline']
    else:
      self.baseline = False 

    #top is just the sum of times avoided words were used? Does loss have to specifically act on word?
    top_shape = ((self.N,))
    top[0].reshape(*top_shape)
      
  def forward(self, bottom, top):
    #This layer does not take words after EOS token into account; consequently a corect sentence will be penalized for generating "red" after it ends.  I think this is okay.

    sampled_words = bottom[0].data #will be size T X N

    avoid_matrix = np.zeros((self.T, self.N))
    for word in self.avoid_words:
      word_idx = self.vocab_dict[word]
      avoid_matrix += (sampled_words == word_idx)

    avoid_reward = -1*np.sum(avoid_matrix, axis=0)

    if self.baseline:
      top[0].data[...] = avoid_reward - np.mean(avoid_reward)
    else:
      top[0].data[...] = avoid_reward

  def reshape(self, bottom, top):
    pass

  def backward(self, top, propagate_down, bottom):
    pass

class sequenceClassificationLoss(caffe.Layer):

  def setup(self, bottom, top):
    #params: vocabulary, which words we should avoid
    params = eval(self.param_str)
    class_probs = bottom[0].data
    length_label = bottom[1].data
    data_label = bottom[2].data

    self.T = class_probs.shape[0]   
    self.N = class_probs.shape[1]  
    if 'baseline' in params.keys():
      self.baseline = params['baseline']
    else:
      self.baseline = False 

    #top is just the probability that a sentence belongs to a certain class 
    top_shape = ((self.N,))
    top[0].reshape(*top_shape)
      
  def forward(self, bottom, top):

    class_probs = bottom[0].data
#    length_label = bottom[1].data
    sampled_sentence = bottom[1].data
    data_label = bottom[2].data

    #class probs is TXNX100; class predictions should be TXN
    reward = np.zeros((self.N,))
    for n, l in enumerate(data_label):
      end_sent_array = np.where(sampled_sentence[:,n] == 0)
      if len(end_sent_array[0]) > 0:
        end_sent = np.where(sampled_sentence[:,n] == 0)[0][0]
      else:
        end_sent = 19
      reward[n] = class_probs[end_sent,n,int(l)]


    if self.baseline:
      top[0].data[...] = reward - np.mean(reward)
    else:
      top[0].data[...] = reward

  def reshape(self, bottom, top):
    pass

  def backward(self, top, propagate_down, bottom):
    pass

