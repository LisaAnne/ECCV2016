#Code by Lisa Anne Hendricks for "Generating Visual Explanations"
#
#@article{hendricks2016generating,
#  title={Generating Visual Explanations},
#  author={Hendricks, Lisa Anne and Akata, Zeynep and Rohrbach, Marcus and Donahue, Jeff and Schiele, Bernt and Darrell, Trevor},
#  journal={arXiv preprint arXiv:1603.08507},
#  year={2016}
#}

#Makes LRCN and unrolled LRCN networks and classifier

from __future__ import print_function
import sys
sys.path.append('utils/')
try: 
  from init import *
except:
  print("Please update utils/init.py to reflect your environment.  Copy utils/init.example.py to utils/init.py and update to match the paths on your machine")
  sys.exit()
sys.path.insert(0, caffe_dir + 'python/')
import caffe
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2
import pdb
from caffe_net import *

class lrcn(caffe_net):

  def __init__(self, data_inputs, lstm_dim = 1000, embed_dim = 1000, class_conditional=True, image_conditional=True, class_size=200, image_dim=8192):
    self.n = caffe.NetSpec()
    self.silence_count = 0
    self.data_inputs = data_inputs
    self.param_str = data_inputs['param_str']
    vocab_txt = open_txt(self.param_str['vocabulary'])
    self.vocab_size = len(vocab_txt) + 1 #+1 for EOS 
    self.lstm_dim = int(embed_dim) 
    self.embed_dim = int(lstm_dim)
    self.cc = class_conditional #class_conditional
    self.ic = image_conditional #image_conditional
    self.class_size = class_size
    self.image_dim = image_dim
    self.feature_dim = self.image_dim + self.class_size 

  def make_lrcn_net_lm(self, bottom_data, bottom_cont, bottom_sent, top_name='predict'):
    self.n.tops['embed'] = self.embed(bottom_sent, self.embed_dim, input_dim=self.vocab_size, bias_term=False, weight_filler=self.uniform_weight_filler(-.08, .08)) 
    
    lstm1 = self.lstm(self.n.tops['embed'], bottom_cont, lstm_hidden=self.lstm_dim)
    setattr(self.n, 'lstm1', lstm1)  

    lstm2 = self.lstm(lstm1, bottom_cont, lstm_static = bottom_data, lstm_hidden=self.lstm_dim)
    setattr(self.n, 'lstm2', lstm2)
    self.n.tops[top_name] = L.InnerProduct(self.n.tops['lstm2'], 
                            num_output=self.vocab_size, axis=2, 
                            weight_filler=self.uniform_weight_filler(-.08, .08), 
                            bias_filler=self.constant_filler(0), 
                            param=self.init_params([[1,1], [2,0]]))
 
  def make_sentence_generation_deploy(self):
    self.n.tops['data'] = self.dummy_data_layer([1,self.image_dim])
    self.n.tops['fc8'] = L.InnerProduct(self.n.tops['data'], num_output=1000, weight_filler=self.constant_filler(0), bias_filler=self.constant_filler(0))
    self.write_net('prototxt/deploy.prototxt')

    self.n = caffe.NetSpec()
    self.n.tops['input_sentence'] = self.dummy_data_layer([1,1000])
    self.n.tops['cont_sentence'] = self.dummy_data_layer([1,1000])
    self.n.tops['image_features'] = self.dummy_data_layer([1000, 1000])
    self.make_lrcn_net_lm(self.n.tops['image_features'], self.n.tops['cont_sentence'], self.n.tops['input_sentence'], 'predict')
    self.n.tops['probs'] = self.softmax(self.n.tops['predict'], axis=2) 
    self.write_net('prototxt/wtd_1000.prototxt')

    self.n = caffe.NetSpec()
    self.n.tops['input_sentence'] = self.dummy_data_layer([1,1000])
    self.n.tops['cont_sentence'] = self.dummy_data_layer([1,1000])
    self.n.tops['image_features'] = self.dummy_data_layer([1000, 2000])
    self.make_lrcn_net_lm(self.n.tops['image_features'], self.n.tops['cont_sentence'], self.n.tops['input_sentence'], 'predict')
    self.n.tops['probs'] = self.softmax(self.n.tops['predict'], axis=2) 
    self.write_net('prototxt/wtd_2000.prototxt')
    self.n = caffe.NetSpec()
    self.n.tops['input_sentence'] = self.dummy_data_layer([20,1000])
    self.n.tops['cont_sentence'] = self.dummy_data_layer([20,1000])
    self.n.tops['image_features'] = self.dummy_data_layer([1000, 2000])
    self.make_lrcn_net_lm(self.n.tops['image_features'], self.n.tops['cont_sentence'], self.n.tops['input_sentence'], 'predict')
    self.n.tops['probs'] = self.softmax(self.n.tops['predict'], axis=2) 
    self.write_net('prototxt/wtd_1000_20words.prototxt')
 
  def make_sentence_generation_net(self, save_file, accuracy=False, loss=True):
    self.n = caffe.NetSpec()
    data_inputs = self.data_inputs
    param_str = self.param_str

    data_tops = self.python_input_layer(data_inputs['module'], data_inputs['layer'], param_str)
    self.rename_tops(data_tops, data_inputs['param_str']['top_names'])
    self.silence(self.n.tops[param_str['data_label']])
 
    assert self.cc | self.ic

    if self.ic:
      feature_name = 'fc8'
      self.n.tops[feature_name] = L.InnerProduct(self.n.tops['image_data'], num_output=1000, weight_filler=self.uniform_weight_filler(-.08, .08), bias_filler=self.constant_filler(0), param=self.init_params([[1,1], [2,0]]))

    if (self.cc) & (not self.ic):
       lrcn_input = 'data_label_feat'
       self.silence(self.n.tops['image_data'])
    elif (not self.cc) & (self.ic):
       lrcn_input = feature_name
       self.silence(self.n.tops['data_label_feat'])
    else:
      self.n.tops['explanation_input'] = L.Concat(self.n.tops[feature_name], self.n.tops['data_label_feat'], axis=1)
      lrcn_input= 'explanation_input'

    self.make_lrcn_net_lm(self.n.tops[lrcn_input], self.n.tops[param_str['text_marker_key']], self.n.tops[param_str['text_data_key']], 'predict')
 
    if loss:
      self.n.tops['loss'] = self.softmax_loss(self.n.tops['predict'], self.n.tops[param_str['text_label_key']], loss_weight=20, axis=2)
    if accuracy:
      self.n.tops['accuracy'] = self.accuracy(self.n.tops['predict'], self.n.tops[param_str['text_label_key']], axis=2)
    self.write_net(save_file)
  
  def caption_classifier(self, save_file, accuracy=False, loss=True, deploy=False, embed_drop=0, lstm_drop=0):
    self.n = caffe.NetSpec()
    data_inputs = self.data_inputs
    param_str = self.param_str

    assert not (loss & deploy) #having loss in deploy net does not make much sense...

    if deploy:
      self.n.tops['input_sentence'] = self.dummy_data_layer([20,100])
      self.n.tops['cont_sentence'] = self.dummy_data_layer([20,100])
      if accuracy:
        self.n.tops['data_label'] = self.dummy_data_layer([1,1000])
    else:
      data_tops = self.python_input_layer(data_inputs['module'], data_inputs['layer'], param_str)
      self.rename_tops(data_tops, data_inputs['param_str']['top_names'])
      self.silence(self.n.tops['target_sentence'])

    embed_name = 'embed'
    self.n.tops[embed_name] = self.embed(self.n.tops['input_sentence'], self.embed_dim, input_dim=self.vocab_size, bias_term=False, weight_filler=self.uniform_weight_filler(-.08, .08)) 

    if (embed_drop > 0) & (not deploy):
      self.n.tops['embed-drop'] = L.Dropout(self.n.tops[embed_name], dropout_ratio=embed_drop)
      embed_name = 'embed-drop' 
 
    lstm_name = 'lstm'
    self.n.tops[lstm_name] = self.lstm(self.n.tops[embed_name], self.n.tops['cont_sentence'], lstm_hidden=self.lstm_dim)
    if (lstm_drop > 0) & (not deploy):
      self.n.tops['lstm-drop'] = L.Dropout(self.n.tops[lstm_name], dropout_ratio=lstm_drop, in_place=True) 
      lstm_name = 'lstm-drop' 
 
    self.n.tops['predict'] = L.InnerProduct(self.n.tops[lstm_name], 
                            num_output=200, axis=2, 
                            weight_filler=self.uniform_weight_filler(-.08, .08), 
                            bias_filler=self.constant_filler(0), 
                            param=self.init_params([[1,1], [2,0]]))
  
    if loss:
      self.n.tops['loss'] = self.softmax_loss(self.n.tops['predict'], self.n.tops['data_label'], loss_weight=1, axis=2)
    if accuracy:
      self.n.tops['accuracy'] = self.accuracy(self.n.tops['predict'], self.n.tops['data_label'], axis=2)
    if deploy:
      self.n.tops['probs'] = self.softmax(self.n.tops['predict'], axis=2)
    self.write_net(save_file)
