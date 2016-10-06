from __future__ import print_function
from init import *
import sys
sys.path.insert(0, caffe_dir + 'python/')
import caffe
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2
import pdb
from caffe_net import *
import lrcn

class reinforce(lrcn.lrcn):
  
  #reinforce nets will just take lrcn nets and add in sampling to the last layer   
 
  def __init__(self, data_inputs, lstm_dim = 1000, embed_dim = 1000, cc=True, class_size=1000, image_dim=1000, baseline=False, separate_sents=False, T=20):
    self.n = caffe.NetSpec()
    self.silence_count = 0
    self.data_inputs = data_inputs
    self.param_str = data_inputs['param_str']
    vocab_txt = open_txt(self.param_str['vocabulary'])
    self.vocab_size = len(vocab_txt) + 1 #+1 for EOS 
    self.lstm_dim = 1000
    self.gate_dim = self.lstm_dim*4 
    self.embed_dim = 1000
    self.cc = cc #class_conditional
    self.class_size = class_size
    self.image_dim = image_dim
    self.feature_dim = self.image_dim + self.class_size 
    if 'stream_size' in self.param_str.keys():
      self.T = self.param_str['stream_size']
    else:
      self.T = T 
    if 'batch_size' in self.param_str.keys():
      self.N = self.param_str['batch_size']
    else:
      self.N = 100
    self.baseline = baseline
    self.separate_sents = separate_sents 

  def make_caption_model(self, static_input='fc8'):

    weight_filler = self.uniform_weight_filler(-0.08, 0.08)
    embed_learning_param = self.named_params(['embed_w'], [[1,1]]) 
    predict_learning_param = self.named_params(['predict_w', 'predict_b'], [[1,1],[2,0]]) 
    #xstatic
    self.n.tops['x_static_transform'] = L.InnerProduct(self.n.tops[static_input], num_output=self.gate_dim,
        bias_term=False, weight_filler=weight_filler,
        param=[dict(lr_mult=1, decay_mult=1, name='W_xc_x_static')])
    lstm_static = 'x_static_transform_reshape'
    self.n.tops[lstm_static] = L.Reshape(self.n.tops['x_static_transform'], shape=dict(dim=[1, -1, self.gate_dim]))
     
    for t in range(self.T):
      bottom_sent = 'input_sent_%d' %t
      bottom_cont_t = 'bottom_cont_%d' %t
      bottom_cont_reshape_t = 'bottom_cont_reshape_%d' %t
      top_name_predict = 'predict_%d' %t
      top_name_probs = 'probs_%d' %t
      top_name_probs_reshape = 'probs_reshape_%d' %t
      top_name_prev_hidden1 = 'lstm1_h%d' %t
      top_name_prev_cell1 = 'lstm1_c%d' %t
      top_name_prev_hidden2 = 'lstm2_h%d' %t
      top_name_prev_cell2 = 'lstm2_c%d' %t
      top_name_hidden1 = 'lstm1_h%d' %(t+1)
      top_name_cell1 = 'lstm1_c%d' %(t+1)
      top_name_hidden2 = 'lstm2_h%d' %(t+1)
      top_name_cell2 = 'lstm2_c%d' %(t+1)
      
      self.n.tops[bottom_cont_reshape_t] = L.Reshape(self.n.tops[bottom_cont_t], shape=dict(dim=[1,1,-1]))

      self.n.tops['embed_%d' %t] = self.embed(self.n.tops[bottom_sent], self.embed_dim, input_dim=self.vocab_size, bias_term=False, weight_filler=self.uniform_weight_filler(-.08, .08), learning_param=embed_learning_param, propagate_down=[0]) 
      
      self.n.tops[top_name_hidden1], self.n.tops[top_name_cell1] = self.lstm_unit('lstm1', self.n.tops['embed_%d' %t],
                                                                   self.n.tops[bottom_cont_reshape_t],  
                                                                   h=self.n.tops[top_name_prev_hidden1], c=self.n.tops[top_name_prev_cell1], 
                                                                   batch_size=self.N, timestep=t,
                                                                   lstm_hidden=self.lstm_dim)
  
      self.n.tops[top_name_hidden2], self.n.tops[top_name_cell2] = self.lstm_unit('lstm2', self.n.tops[top_name_hidden1], 
                                                                   self.n.tops[bottom_cont_reshape_t], static=self.n.tops[lstm_static], 
                                                                   h=self.n.tops[top_name_prev_hidden2], c=self.n.tops[top_name_prev_cell2], 
                                                                   batch_size=self.N, timestep=t,
                                                                   lstm_hidden=self.lstm_dim)

      self.n.tops[top_name_predict] = L.InnerProduct(self.n.tops[top_name_hidden2], 
                              num_output=self.vocab_size, axis=2, 
                              weight_filler=self.uniform_weight_filler(-.08, .08), 
                              bias_filler=self.constant_filler(0), 
                              param=predict_learning_param)
      self.n.tops[top_name_probs] = self.softmax(self.n.tops[top_name_predict], axis=2)
      self.n.tops[top_name_probs_reshape] = L.Reshape(self.n.tops[top_name_probs], shape=dict(dim=[-1, self.vocab_size]))
      self.n.tops['word_sample_%d' %t] = L.Sample(self.n.tops[top_name_probs_reshape], propagate_down=[0])
      self.n.tops['word_sample_reshape_%d' %(t+1)] = L.Reshape(self.n.tops['word_sample_%d' %t], shape=dict(dim=[1,-1]))
      if self.separate_sents:
        #silence the last half of the input sentences
        sample_sents = L.Slice(self.n.tops['word_sample_reshape_%d' %(t+1)], axis=1, slice_point=[self.slice_point], ntop=2)
        self.rename_tops(sample_sents, ['reg_word_sample_reshape_%d' %(t+1), 'rl_word_sample_reshape_%d' %(t+1)])
        self.silence(self.n.tops['reg_word_sample_reshape_%d' %(t+1)])
        if t < self.T-1:
          self.n.tops['input_sent_%d' %(t+1)] = L.Concat(self.n.tops['reg_input_sent_%d' %(t+1)], self.n.tops['rl_word_sample_reshape_%d' %(t+1)], axis=1) 
    self.silence([self.n.tops['lstm1_c%d' %(self.T)]])
    self.silence([self.n.tops['lstm2_c%d' %(self.T)]])

  def lrcn_reinforce(self, save_name, RL_loss='lstm_classification', lw=20):
   
    data_inputs = self.data_inputs
    param_str = self.param_str

    ss_tag = 'reg_'
    #reg sentences will be the first part of the batch
    if self.separate_sents:
      if not 'batch_size' in param_str.keys():
        param_str['batch_size'] = 100
      self.slice_point = param_str['batch_size']/2
      self.batch_size = param_str['batch_size']
 
    param_str_loss = {}
    param_str_loss['vocab'] = param_str['vocabulary']
    param_str_loss['avoid_words'] = ['red', 'small']
    if self.baseline:
      param_str_loss['baseline'] = True
    data_input = 'fc8'

    data_tops = self.python_input_layer(data_inputs['module'], data_inputs['layer'], param_str)
    self.rename_tops(data_tops, data_inputs['param_str']['top_names'])
    feature_name = 'fc8'
    self.n.tops[feature_name] = L.InnerProduct(self.n.tops[param_str['image_data_key']], num_output=1000, weight_filler=self.uniform_weight_filler(-.08, .08), bias_filler=self.constant_filler(0), param=self.learning_params([[1,1], [2,0]]))

    if self.cc:
      #If class conditional
      data_top = self.n.tops['fc8']
      class_top = self.n.tops[param_str['data_label_feat']]
      self.n.tops['class_input'] = L.Concat(data_top, class_top, axis=1)
      data_input = 'class_input' 
    else:
      self.silence(self.n.tops[param_str['data_label_feat']])
 
    bottom_sent = self.n.tops[param_str['text_data_key']]
    bottom_cont = self.n.tops[param_str['text_marker_key']]

    #prep for caption model
    bottom_cont_slice = L.Slice(bottom_cont, ntop=self.T, axis=0)
    self.rename_tops(bottom_cont_slice, ['bottom_cont_%d' %i for i in range(self.T)])

    if not self.separate_sents:
      bottom_sent_slice = L.Slice(bottom_sent, ntop=self.T, axis=0)
      self.rename_tops(bottom_sent_slice, ['input_sent_%d' %i for i in range(self.T)])
      target_sentence = self.n.tops['target_sentence']
    else:
      bottom_sents = L.Slice(bottom_sent, slice_point = [self.slice_point], axis=1, ntop=2)
      self.rename_tops(bottom_sents, ['reg_input_sent', 'rl_input_sent'])
      reg_bottom_sents_slice = L.Slice(self.n.tops['reg_input_sent'], axis=0, ntop=20)
      rl_bottom_sents_slice = L.Slice(self.n.tops['rl_input_sent'], axis=0, ntop=20)
      self.silence([rl_bottom_sents_slice[i] for i in range(1, self.T)])
      self.n.tops['input_sent_0'] = L.Concat(reg_bottom_sents_slice[0], rl_bottom_sents_slice[0], axis=1)
      self.rename_tops(reg_bottom_sents_slice, ['reg_input_sent_%d' %i for i in range(1,self.T)])

      self.rename_tops(reg_bottom_sents_slice, ['reg_input_sent_%d' %i for i in range(self.T)])
      slice_target_sentence = L.Slice(self.n.tops['target_sentence'], slice_point = [self.slice_point], axis=1, ntop=2)
      self.rename_tops(slice_target_sentence, ['reg_target_sentence', 'rl_target_sentence'])
      self.silence(self.n.tops['rl_target_sentence'])
      target_sentence = self.n.tops['reg_target_sentence'] 
 
    self.n.tops['lstm1_h0'] = self.dummy_data_layer([1, self.N, self.lstm_dim], 0)
    self.n.tops['lstm1_c0'] = self.dummy_data_layer([1, self.N, self.lstm_dim], 0)
    self.n.tops['lstm2_h0'] = self.dummy_data_layer([1, self.N, self.lstm_dim], 0)
    self.n.tops['lstm2_c0'] = self.dummy_data_layer([1, self.N, self.lstm_dim], 0)

    self.make_caption_model(static_input=data_input)
   
    #prep bottoms for loss
    predict_tops = [self.n.tops['predict_%d' %i] for i in range(self.T)]
    self.n.tops['predict_concat'] = L.Concat(*predict_tops, axis=0)
    if self.separate_sents:
      word_sample_tops = [self.n.tops['rl_word_sample_reshape_%d' %i] for i in range(1,self.T+1)]
      self.n.tops['word_sample_concat'] = L.Concat(*word_sample_tops, axis=0)
      concat_predict_tops = L.Slice(self.n.tops['predict_concat'], slice_point=[self.slice_point], axis=1, ntop=2)
      reg_predict = concat_predict_tops[0]
      RL_predict = concat_predict_tops[1]
      bottom_cont_tops = L.Slice(bottom_cont, slice_point=[self.slice_point], axis=1, ntop=2)
      self.silence(bottom_cont_tops[0])
      label_tops = L.Slice(self.n.tops[param_str['data_label']], slice_point=[self.slice_point], axis=0, ntop=2)
      self.silence(label_tops[0])
      self.rename_tops([bottom_cont_tops[1], label_tops[1]], ['rl_bottom_cont', 'rl_label_top'])
      label_top = self.n.tops['rl_label_top']
      bottom_cont = self.n.tops['rl_bottom_cont'] 
    else:
      word_sample_tops = [self.n.tops['word_sample_reshape_%d' %i] for i in range(1,self.T+1)]
      self.n.tops['word_sample_concat'] = L.Concat(*word_sample_tops, axis=0)
      reg_predict = self.n.tops['predict_concat']
      RL_predict = self.n.tops['predict_concat']
      label_top = self.n.tops[param_str['data_label']]

    #RL loss
    if RL_loss == 'lstm_classification':
      self.n.tops['embed_classification'] = self.embed(self.n.tops['word_sample_concat'], 1000, input_dim=self.vocab_size, bias_term=False, learning_param=self.learning_params([[0,0]]))
      self.lstm(self.n.tops['embed_classification'], bottom_cont, top_name='lstm_classification', learning_param_lstm=self.learning_params([[0,0],[0,0],[0,0]]), lstm_hidden=1000)  
      self.n.tops['predict_classification'] = L.InnerProduct(self.n.tops['lstm_classification'], num_output=200, axis=2)    
      self.n.tops['probs_classification'] = L.Softmax(self.n.tops['predict_classification'], axis=2)    
      #classification reward layer: classification, word_sample_concat (to get sentence length), 
      #data label should be single stream; even though trained with 20 stream...
      self.n.tops['reward'] = self.python_layer([self.n.tops['probs_classification'], self.n.tops['word_sample_concat'], label_top], 'loss_layers', 'sequenceClassificationLoss', param_str_loss) 

    self.n.tops['reward_reshape'] = L.Reshape(self.n.tops['reward'], shape = dict(dim=[1,-1]))
    self.n.tops['reward_tile'] = L.Tile(self.n.tops['reward_reshape'], axis=0, tiles=self.T)

   #softmax with sampled words as "correct" word
    self.n.tops['sample_loss'] = self.softmax_per_inst_loss(RL_predict, self.n.tops['word_sample_concat'], axis=2)
    self.n.tops['sample_reward'] = L.Eltwise(self.n.tops['sample_loss'], self.n.tops['reward_tile'], propagate_down=[1,0], operation=0)
    avoid_lw = 100
    self.n.tops['normalized_reward'] = L.Power(self.n.tops['sample_reward'], scale=(1./self.N)*avoid_lw)
    self.n.tops['sum_rewards'] = L.Reduction(self.n.tops['normalized_reward'], loss_weight=[1])                                    
    self.n.tops['sentence_loss'] = self.softmax_loss(reg_predict, target_sentence, axis=2, loss_weight=20)

    self.write_net(save_name)

  def lrcn_reinforce_im_deploy(self, save_name):
    self.n.tops['data'] = self.dummy_data_layer([self.N, 3, 227, 227])
    self.make_lrcn_caffenet_image(self.n.tops['data'], 'fc8')
    self.write_net(save_name)
  
  def lrcn_reinforce_wtd_deploy(self, save_name):
    self.n.tops['input_sentence'] = self.dummy_data_layer([1, self.N])
    self.n.tops['cont_sentence'] = self.dummy_data_layer([1, self.N])
    self.n.tops['image_features'] = self.dummy_data_layer([self.N, self.feature_dim])
    self.n.tops['lstm1_h0'] = self.dummy_data_layer([1, self.N, self.lstm_dim])
    self.n.tops['lstm1_c0'] = self.dummy_data_layer([1, self.N, self.lstm_dim])
    self.n.tops['lstm2_h0'] = self.dummy_data_layer([1, self.N, self.lstm_dim])
    self.n.tops['lstm2_c0'] = self.dummy_data_layer([1, self.N, self.lstm_dim])
    self.n.tops['input_sent_0'] = L.Split(self.n.tops['input_sentence'])
    self.n.tops['bottom_cont_0'] = L.Split(self.n.tops['cont_sentence'])

    self.make_caption_model(static_input='image_features')
    self.n.tops['probs'] = L.Split(self.n.tops['probs_0'])
    self.n.tops['probs'] = L.Split(self.n.tops['probs_0'])
    self.write_net(save_name)

