''' TODO: Clean and make more consistent '''

from __future__ import print_function
from init import *
import sys
sys.path.insert(0, caffe_dir + 'python/')
import caffe
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2
import pdb
import os
import stat

base_prototxt_file = 'prototxt/'

class caffe_net(object):

  def __init__(self):
    self.n = caffe.NetSpec()
    self.silence_count = 0

  def uniform_weight_filler(self, min_value, max_value):
    return dict(type='uniform', min=min_value, max=max_value)

  def constant_filler(self, value=0):
    return dict(type='constant', value=value)
  
  def gaussian_filler(self, value=0.01):
    return dict(type='gaussian', value=value)

  def write_net(self, save_file):
    write_proto = self.n.to_proto()
    with open(save_file, 'w') as f:
      print(write_proto, file=f)
    print("Wrote net to: %s." %save_file)
    #reinitialized network
    self.__init__()
  
  def init_params(self, param_list, name_list=None):
    if name_list:
      assert len(name_list) == len(param_list)

    param_dicts = []
    for i, param in enumerate(param_list):
      param_dict = {}
      param_dict['lr_mult'] = param[0]
      if len(param) > 1:
        param_dict['decay_mult'] = param[1]
      param_dicts.append(param_dict)
      if name_list:
        param_dict['name'] = name_list[i]

    return param_dicts

  def init_fillers(self, kwargs, weight_filler=None, bias_filler=None, learning_param=None):
    if weight_filler: kwargs['weight_filler'] = weight_filler
    if bias_filler: kwargs['bias_filler'] = bias_filler
    if learning_param: kwargs['learning_param'] = learning_param
    return kwargs 

  #The following methods will return layers.  Input is bottom to layer + parameters, and output is top(s)
  def conv_relu(self, bottom, ks, nout, 
                stride=1, pad=0, group=1, weight_filler=None, bias_filler=None, learning_param=None):

    kwargs = {'kernal_size': ks, 'num_output': nout, 'stride': stride,
               'pad': pad, 'group': group}
    kwargs = init_fillers(kwargs, weight_filler, bias_filler, learning_param)

    conv = L.Convolution(bottom, **kwargs)
    return conv, L.ReLU(conv, in_place=True)

  def fc_relu(self, bottom, nout, 
              weight_filler=None, bias_filler=None, learning_param=None):

    kwargs = {'num_output': nout}
    kwargs = init_fillers(kwargs, weight_filler, biass_filler, learning_param)

    fc = L.InnerProduct(bottom, **kwargs)
    return fc, L.ReLU(fc, in_place=True)

  def sum(self, bottoms):
    return L.Eltwise(*bottoms, operation=1) 

  def subtract(self, bottoms):
    assert len(bottoms) == 2
    negate = L.Power(bottoms[1], scale=-1)
    return L.Eltwise(bottoms[0], bottoms[1], operation=1) 

  def prod(self, bottoms):
    return L.Eltwise(*bottoms, operation=0) 

  def embed(self, bottom, nout, input_dim=8801, bias_term=True, axis=1, propagate_down=False, 
            weight_filler=None, bias_filler=None, learning_param=None):

    kwargs = {'num_output': nout, 'input_dim': input_dim, 'bias_term': bias_term}
    kwargs = self.init_fillers(kwargs, weight_filler, bias_filler, learning_param)
    return L.Embed(bottom, **kwargs)

  def max_pool(self, bottom, ks, stride=1):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

  def accuracy(self, bottom_data, bottom_label, axis=1, ignore_label=-1):
    return L.Accuracy(bottom_data, bottom_label, axis=axis, ignore_label=ignore_label)

  def softmax_loss(self, bottom_data, bottom_label, axis=1, ignore_label=-1, loss_weight=1):
    return L.SoftmaxWithLoss(bottom_data, bottom_label, loss_weight=[loss_weight], loss_param=dict(ignore_label=ignore_label), softmax_param=dict(axis=axis))
 
  def sigmoid_loss(self, bottom_data, bottom_label, loss_weight=1):
    return L.SigmoidCrossEntropyLoss(bottom_data, bottom_label, loss_weight=loss_weight, loss_param=dict(ignore_label=-1))
 
  def sigmoid(self, bottom_data):
    return L.Sigmoid(bottom_data)

  def softmax_per_inst_loss(self, bottom_data, bottom_label, axis=1, ignore_label=-1, loss_weight=0):
    return L.SoftmaxPerInstLoss(bottom_data, bottom_label, loss_weight=[loss_weight], loss_param=dict(ignore_label=ignore_label), softmax_param=dict(axis=axis))

  def softmax(self, bottom_data, axis=1):
    return L.Softmax(bottom_data, axis=axis)

  def python_input_layer(self, module, layer, param_str = {}):
    return L.Python(module=module, layer=layer, param_str=str(param_str), ntop=len(param_str['top_names']))
  
  def python_layer(self, inputs, module, layer, param_str = {}, ntop=1, loss_weight=0):
    return L.Python(*inputs, module=module, layer=layer, param_str=str(param_str), ntop=1, loss_weight=[loss_weight])

  def rename_tops(self, tops, names):
    if not isinstance(tops, tuple): 
      tops = [tops]
    if isinstance(names, str):
      names = [names]
    for top, name in zip(tops, names): setattr(self.n, name, top)

  def dummy_data_layer(self,shape, filler=1):
    #shape should be a list of dimensions
    return L.DummyData(shape=[dict(dim=shape)], data_filler=[self.constant_filler(filler)], ntop=1)

  def silence(self, bottom):
    if isinstance(bottom, list):
      self.n.tops['silence_cell_'+str(self.silence_count)] = L.Silence(*bottom, ntop=0)
    else:
      self.n.tops['silence_cell_'+str(self.silence_count)] = L.Silence(bottom, ntop=0)
    self.silence_count += 1

  def make_caffenet(self, bottom, return_layer, weight_filler={}, bias_filler={}, learning_param={}):
      default_weight_filler = self.gaussian_filler() 
      default_bias_filler = self.gaussian_filler(1)
      default_learning_param = self.init_params([[1,1],[2,0]])
      for layer in ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8']:
        if layer not in weight_filler.keys(): weight_filler[layer] = default_weight_filler
        if layer not in bias_filler.keys(): bias_filler[layer] = default_bias_filler
        if layer not in learning_param.keys(): learning_param[layer] = default_learning_param

      self.n.tops['conv1'], self.n.tops['relu1'] = self.conv_relu(bottom, 11, 96, stride=4,
                                                             weight_filler=weight_filler['conv1'],
                                                             bias_filler=bias_filler['conv1'],
                                                             learning_param=learning_param['conv1'])
      if return_layer in self.n.tops.keys(): return
      self.n.tops['pool1'] = self.max_pool(self.n.tops['relu1'], 3, stride=2)
      if return_layer in self.n.tops.keys(): return
      self.n.tops['norm1'] = L.LRN(self.n.tops['pool1'], local_size=5, alpha=1e-4, beta=0.75)
      if return_layer in self.n.tops.keys(): return

      self.n.tops['conv2'], self.n.tops['relu2'] = self.conv_relu(self.n.tops['norm1'], 5, 256, pad=2, group=2,
                                   weight_filler=weight_filler['conv2'],
                                   bias_filler=bias_filler['conv2'],
                                   learning_param=learning_param['conv2'])
      if return_layer in self.n.tops.keys(): return
      self.n.tops['pool2'] = self.max_pool(self.n.tops['relu2'], 3, stride=2)
      if return_layer in self.n.tops.keys(): return
      self.n.tops['norm2'] = L.LRN(self.n.tops['pool2'], local_size=5, alpha=1e-4, beta=0.75)
      if return_layer in self.n.tops.keys(): return 

      self.n.tops['conv3'], self.n.tops['relu3'] = self.conv_relu(self.n.tops['norm2'], 3, 384, pad=1,
                                                             weight_filler=weight_filler['conv3'],
                                                             bias_filler=bias_filler['conv3'],
                                                             learning_param=learning_param['conv3'])
      if return_layer in self.n.tops.keys(): return 

      self.n.tops['conv4'], self.n.tops['relu4'] = self.conv_relu(self.n.tops['relu3'], 3, 384, pad=1, group=2,
                                                             weight_filler=weight_filler['conv4'],
                                                             bias_filler=bias_filler['conv4'],
                                                             learning_param=learning_param['conv4'])
      if return_layer in self.n.tops.keys(): return 

      self.n.tops['conv5'], self.n.tops['relu5'] = self.conv_relu(self.n.tops['relu4'], 3, 256, pad=1, group=2,
                                                             weight_filler=weight_filler['conv5'],
                                                             bias_filler=bias_filler['conv5'],
                                                             learning_param=learning_param['conv5'])
      if return_layer in self.n.tops.keys(): return 
      self.n.tops['pool5'] = self.max_pool(self.n.tops['relu5'], 3, stride=2)
      if return_layer in self.n.tops.keys(): return 

      self.n.tops['fc6'], self.n.tops['relu6'] = self.fc_relu(self.n.tops['pool5'], 4096,
                                                             weight_filler=weight_filler['fc6'],
                                                             bias_filler=bias_filler['fc6'],
                                                             learning_param=learning_param['fc6'])
      if return_layer in self.n.tops.keys(): return 
      self.n.tops['drop6'] = L.Dropout(self.n.tops['relu6'], in_place=True)
      if return_layer in self.n.tops.keys(): return 
      self.n.tops['fc7'], self.n.tops['relu7'] = self.fc_relu(self.n.tops['drop6'], 4096,
                                                             weight_filler=weight_filler['fc7'],
                                                             bias_filler=bias_filler['fc7'],
                                                             learning_param=learning_param['fc7'])
      if return_layer in self.n.tops.keys(): return 'relu7'
      self.n.tops['drop7'] = L.Dropout(self.n.tops['relu7'], in_place=True)
      if return_layer in self.n.tops.keys(): return  
      self.n.tops['fc8'] = L.InnerProduct(self.n.tops['drop7'], num_output=1000,  
                                          weight_filler=weight_filler['fc8'],
                                          bias_filler=bias_filler['fc8'],
                                          param=learning_param['fc8'])

  
  def lstm(self, data, markers, lstm_static=None, lstm_hidden=1000, weight_filler=None, bias_filler=None, learning_param_lstm=None):

    #default params
    if not weight_filler: weight_filler = self.uniform_weight_filler(-.08, .08)
    if not bias_filler: bias_filler = self.constant_filler(0)
    if not learning_param_lstm: learning_param_lstm = self.init_params([[1,1],[1,1],[1,1]])

    if lstm_static:
      return L.LSTM(data, markers, lstm_static, param=learning_param_lstm,
                 recurrent_param=dict(num_output=lstm_hidden, weight_filler=weight_filler, bias_filler=bias_filler))
    else: 
      return L.LSTM(data, markers, param=learning_param_lstm,
                 recurrent_param=dict(num_output=lstm_hidden, weight_filler=weight_filler, bias_filler=bias_filler))
 
  def lstm_unit(self, prefix, x, cont, static=None, h=None, c=None,
         batch_size=100, timestep=0, lstm_hidden=1000,
         weight_lr_mult=1, bias_lr_mult=2,
         weight_decay_mult=1, bias_decay_mult=0, concat_hidden=True,
         weight_filler=None, bias_filler=None, prefix_layer=None):
 
    #assume static input is already transformed
    if not prefix_layer:
      prefix_layer = prefix

    if not weight_filler:
      weight_filler = self.uniform_weight_filler(-0.08, 0.08)
    if not bias_filler:
      bias_filler = self.constant_filler(0)
    if not h:
      h = self.dummy_data_layer([1, batch_size, lstm_hidden], 1)
    if not c:
      c = self.dummy_data_layer([1, batch_size, lstm_hidden], 1)
    gate_dim=lstm_hidden*4

    def get_layer_name(name):
        return '%s_%s' % (prefix_layer, name)
    def get_weight_name(name):
        return '%s_%s' % (prefix, name)
    def get_param(weight_name, bias_name=None):
        #TODO: write this in terms of earlier method "init_params"
        w = dict(lr_mult=weight_lr_mult, decay_mult=weight_decay_mult,
                 name=get_weight_name(weight_name))
        if bias_name is not None:
            b = dict(lr_mult=bias_lr_mult, decay_mult=bias_decay_mult,
                     name=get_weight_name(bias_name))
            return [w, b]
        return [w]

    # gate_dim is the dimension of the cell state inputs:
    # 4 gates (i, f, o, g), each with dimension dim
    # Add layer to transform all timesteps of x to the hidden state dimension.
    #     x_transform = W_xc * x + b_c
    x = L.InnerProduct(x, num_output=gate_dim, axis=2,
        weight_filler=weight_filler, bias_filler=bias_filler,
        param=get_param('W_xc', 'b_c'))
    self.rename_tops(x, get_layer_name('%d_x_transform' %timestep))

    h_conted = L.Scale(h, cont, axis=0) 
    h = L.InnerProduct(h_conted, num_output=gate_dim, axis=2, bias_term=False,
        weight_filler=weight_filler, param=get_param('W_hc'))
    h_name = get_layer_name('%d_h_transform' %timestep)
    if not hasattr(self.n, h_name):
        setattr(self.n, h_name, h)
    gate_input_args = x, h
    if static is not None:
        gate_input_args += (static, )
    gate_input = L.Eltwise(*gate_input_args)
    assert cont is not None
    c, h = L.LSTMUnit(c, gate_input, cont, ntop=2)
    return h, c 

  def gru_unit(self, prefix, x, cont, static=None, h=None,
         batch_size=100, timestep=0, gru_hidden=1000,
         weight_lr_mult=1, bias_lr_mult=2,
         weight_decay_mult=1, bias_decay_mult=0, concat_hidden=True,
         weight_filler=None, bias_filler=None):

    #assume static input already transformed

    if not weight_filler:
      weight_filler = self.uniform_weight_filler(-0.08, 0.08)
    if not bias_filler:
      bias_filler = self.constant_filler(0)
    if not h:
      h = self.dummy_data_layer([1, batch_size, lstm_hidden], 1)

    def get_name(name):
        return '%s_%s' % (prefix, name)
    def get_param(weight_name, bias_name=None):
        #TODO: write this in terms of earlier method "init_params"
        w = dict(lr_mult=weight_lr_mult, decay_mult=weight_decay_mult,
                 name=get_name(weight_name))
        if bias_name is not None:
            b = dict(lr_mult=bias_lr_mult, decay_mult=bias_decay_mult,
                     name=get_name(bias_name))
            return [w, b]
        return [w]

    gate_dim = gru_hidden*3  

    #transform x_t
    x = L.InnerProduct(x, num_output=gate_dim, axis=2,
        weight_filler=weight_filler, bias_filler=bias_filler,
        param=get_param('W_xc', 'b_c'))
    self.rename_tops(x, get_name('%d_x_transform' %timestep))

    #transform h 
    h_conted = L.Scale(h, cont, axis=0) 
    h = L.InnerProduct(h_conted, num_output=gru_hidden*2, axis=2, bias_term=False,
        weight_filler=weight_filler, param=get_param('W_hc'))
    h_name = get_name('%d_h_transform' %timestep)
    if not hasattr(self.n, h_name):
        setattr(self.n, h_name, h)
    
    #gru stuff TODO: write GRUUnit in caffe?  would make all this much prettier.
    x_transform_z_r, x_transform_hc = L.Slice(x, slice_point=gru_hidden*2, axis=2, ntop=2)
    sum_items = [x_transform_z_r, h]
    if static:
      sum_items += static
    z_r_sum = self.sum(sum_items)
    z_r = L.Sigmoid(z_r_sum)
    z, r = L.Slice(z_r, slice_point=gru_hidden, axis=2, ntop=2)  
    
    z_weighted_h = self.prod([r, h_conted])
    z_h_transform = L.InnerProduct(z_weighted_h, num_output=gru_hidden, axis=2, bias_term=False,
                                   weight_filler=weight_filler, param=get_param('W_hzc'))
    sum_items = [x_transform_hc, z_h_transform]
    if static:
      sum_items += static
    hc_sum = self.sum(sum_items)
    hc = L.TanH(hc)
    
    zm1 = L.Power(z, scale=-1, shift=1)
    h_h = self.prod([zm1, h_conted])
    h_hc = self.prod([z, hc])
    h = self.sum([h_h, h_hc])

    return h 

  def gru(self, prefix, x, cont, static=None, h=None,
          batch_size=100, T=0, gru_hidden=1000,
          weight_lr_mult=1, bias_lr_mult=2, 
          weight_decay_mult=1, bias_decay_mult=0,
          weight_filler=None, bias_filler=None): 

    if not weight_filler:
      weight_filler = self.uniform_weight_filler(-0.08, 0.08)
    if not bias_filler:
      bias_filler = self.constant_filler(0)
    if not h:
      h = self.dummy_data_layer([1, batch_size, lstm_hidden], 1)

    gate_dim = gru_hidden*3  
    if static: #assume static NXF blob
      static_transform = L.InnerProduct(static, num_output=gate_dim, axis=2,
                               weight_filler=weight_filler, bias_filler=bias_filler)
      static_transform = L.Reshape(static, shape=dict(dim=[1,-1,gate_dim]))
      self.rename_tops(static_transform, '%s_x_static' %prefix)
      
    h = None

    x_in = L.Slice(x, ntop=self.T, axis=0)
    cont_in = L.Slice(cont, ntop=self.T, axis=0)

    for t in range(T):
      h = self.gru_unit(prefix, x_in[t], cont[t], static, h,
                        batch_size=batch-size, timestep=t, gru_hidden=gru_hidden,
                        weight_lr_mult=weight_lr_mult, bias_lr_mult=bias_lr_mult,
                        weight_decay_mult=weight_dicay_mult, bias_decay_mult=bias_decay_mult,
                        weight_filler=weight_filler, bias_filler=bias_filler)

    return h

def make_solver(save_name, tag, train_nets, test_nets, **kwargs):

  #set default values
  parameter_dict = kwargs
  if 'test_iter' not in parameter_dict.keys(): parameter_dict['test_iter'] = 10
  if 'test_interval' not in parameter_dict.keys(): parameter_dict['test_interval'] = 1000
  if 'base_lr' not in parameter_dict.keys(): parameter_dict['base_lr'] = 0.01
  if 'lr_policy' not in parameter_dict.keys(): parameter_dict['lr_policy'] = '"step"' 
  if 'display' not in parameter_dict.keys(): parameter_dict['display'] = 10
  if 'max_iter' not in parameter_dict.keys(): parameter_dict['max_iter'] = 30000
  if 'gamma' not in parameter_dict.keys(): parameter_dict['gamma'] = 0.5
  if 'stepsize' not in parameter_dict.keys(): parameter_dict['stepsize'] = 10000
  if 'snapshot' not in parameter_dict.keys(): parameter_dict['snapshot'] = 5000
  if 'momentum' not in parameter_dict.keys(): parameter_dict['momentum'] = 0.9
  if 'weight_decay' not in parameter_dict.keys(): parameter_dict['weight_decay'] = 0.0
  if 'solver_mode' not in parameter_dict.keys(): parameter_dict['solver_mode'] = 'GPU'
  if 'random_seed' not in parameter_dict.keys(): parameter_dict['random_seed'] = 1701
  if 'average_loss' not in parameter_dict.keys(): parameter_dict['average_loss'] = 100
  if 'clip_gradients' not in parameter_dict.keys(): parameter_dict['clip_gradients'] = 10
  #parameter_dict['debug_info'] = 'true'
  #if 'snapshot_format' not in parameter_dict.keys(): parameter_dict['snapshot_format'] ='HDF5'

  snapshot_prefix = 'snapshots/%s' %tag
  parameter_dict['snapshot_prefix'] = '"%s"' %snapshot_prefix
 
  write_txt = open(save_name, 'w')
  for tn in train_nets:
    write_txt.writelines('train_net: "%s"\n' %tn)
  for tn in test_nets:
    write_txt.writelines('test_net: "%s"\n' %tn)
    write_txt.writelines('test_iter: %d\n' %parameter_dict['test_iter'])
  if len(test_nets) > 0:
    write_txt.writelines('test_interval: %d\n' %parameter_dict['test_interval'])

  parameter_dict.pop('test_iter')
  parameter_dict.pop('test_interval')

  for key in parameter_dict.keys():
    write_txt.writelines('%s: %s\n' %(key, parameter_dict[key]))

  write_txt.close()

  print("Wrote solver to %s." %save_name)

def make_bash_script(save_bash, solver, weights=None, gpu=0):
  write_txt = open(save_bash, 'w')
  
  write_txt.writelines('#!/usr/bin/env bash\n\n')
  write_txt.writelines('GPU_ID=%d\n' %gpu)
  if weights:
    write_txt.writelines('WEIGHTS=%s\n\n' %weights)
  write_txt.writelines("export PYTHONPATH='utils/python_layers/:$PYTHONPATH'\n\n")
  if weights:
    write_txt.writelines("%s/build/tools/caffe train -solver %s -weights %s -gpu %d" %(caffe_dir, solver, weights, gpu))
  else:
    write_txt.writelines("%s/build/tools/caffe train -solver %s -gpu %d" %(caffe_dir, solver, gpu))
  write_txt.close()  

  print("Wrote bash scripts to %s." %save_bash)

  #make bash script executable
  st = os.stat(save_bash)
  os.chmod(save_bash, st.st_mode | stat.S_IEXEC)
