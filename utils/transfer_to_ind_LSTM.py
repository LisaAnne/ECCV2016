#Transfer weights from model trained with normal LSTM unit to unrolled LSTM units

import sys
sys.path.append('utils/')
from init import *
import pdb
sys.path.insert(0, pycaffe_path)
sys.path.insert(0, 'utils/python_layers/')
import caffe
caffe.set_mode_gpu()
caffe.set_device(0)
import argparse
import numpy as np

def transfer_net_weights(orig_model, orig_model_weights, new_model):
 
  new_model_weights_save = 'snapshots/' + orig_model_weights.split('/')[-1].split('.')[0] + 'indLSTM.caffemodel'
 
  orig_net = caffe.Net(orig_model, orig_model_weights, caffe.TRAIN)
  new_net = caffe.Net(new_model, caffe.TRAIN)

  for layer in list(set(orig_net.params.keys()) & set(new_net.params.keys())):
    for ix in range(len(new_net.params[layer])):
      new_net.params[layer][ix].data[...] = orig_net.params[layer][ix].data

  embed_weights = orig_net.params['embed'][0].data
  lstm1_W_xc = orig_net.params['lstm1'][0].data
  lstm1_b_c = orig_net.params['lstm1'][1].data
  lstm1_W_hc = orig_net.params['lstm1'][2].data
  lstm2_W_xc = orig_net.params['lstm2'][0].data
  lstm2_b_c = orig_net.params['lstm2'][1].data
  lstm2_W_x_static = orig_net.params['lstm2'][2].data
  lstm2_W_hc = orig_net.params['lstm2'][3].data
  predict_w = orig_net.params['predict'][0].data
  predict_b = orig_net.params['predict'][1].data

  new_net.params['embed_0'][0].data[...] = embed_weights
  new_net.params['x_static_transform'][0].data[...] = lstm2_W_x_static
  new_net.params['lstm1_0_x_transform'][0].data[...] = lstm1_W_xc
  new_net.params['lstm1_0_x_transform'][1].data[...] = lstm1_b_c
  new_net.params['lstm1_0_h_transform'][0].data[...] = lstm1_W_hc
  new_net.params['lstm2_0_x_transform'][0].data[...] = lstm2_W_xc
  new_net.params['lstm2_0_x_transform'][1].data[...] = lstm2_b_c
  new_net.params['lstm2_0_h_transform'][0].data[...] = lstm2_W_hc
  new_net.params['predict_0'][0].data[...] = predict_w
  new_net.params['predict_0'][1].data[...] = predict_b

  for layer in new_net.params.keys():
    for ix in range(len(new_net.params[layer])):
       print layer, ix, np.max(new_net.params[layer][ix].data)

  new_net.save(new_model_weights_save)
  print "New model saved to %s." %new_model_weights_save
  return new_model_weights_save

def transfer_combine_weights(model, classify_model, caption_weights, classifier_weights):
  net = caffe.Net(model, caption_weights, caffe.TRAIN)
  classify_net = caffe.Net(classify_model, classifier_weights, caffe.TRAIN)

  for param in classify_net.params.keys():
    param_new = '%s_classification' %param
    for i in range(len(classify_net.params[param])):
      net.params[param_new][i].data[...] = classify_net.params[param][i].data

  orig_snap_tag = caption_weights.split('/')[-1].split('.caffemodel')[0]
  classify_snap_tag = classifier_weights.split('/')[-1].split('.caffemodel')[0]
  new_net_save = 'snapshots/%s_%s.caffemodel' %(orig_snap_tag, classify_snap_tag)
  net.save(new_net_save)
  
  for layer in net.params.keys():
    for ix in range(len(net.params[layer])):
       print layer, ix, np.max(net.params[layer][ix].data)
  print "Saved caffemodel to %s." %new_net_save 
  return new_net_save

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--orig_model",type=str, default='prototxt/lrcn_freeze.prototxt')
  parser.add_argument("--orig_model_weights",type=str, default='snapshots/birds_from_scratch_fgSplit_freezeConv_iter_20000.caffemodel')
  parser.add_argument("--new_model",type=str, default='prototxt/lrcn_unroll_lstm_train.prototxt')

  parser.add_argument("--combine_model", type=str, default='prototxt/lrcn_reinforce_lstm_classification_sentenceLoss_wbT_train.prototxt')
  #parser.add_argument("--classify_model", type=str, default='eccv_prototxts/caption_classifier_embedDrop_75_lstmDrop_90_embedHidden_1000_lstmHidden_1000_train.prototxt')
  parser.add_argument("--classify_model", type=str, default='prototxt/caption_classifier_embedDrop_75_lstmDrop_75_embedHidden_1000_lstmHidden_1000_train.prototxt')
  parser.add_argument("--caption_weights", type=str, default='snapshots/birds_from_scratch_fgSplit_freezeConv_iter_20000indLSTM.caffemodel')
  #parser.add_argument("--classify_weights", type=str, default='eccv_snapshots/caption_classifier_embedDrop_75_lstmDrop_75_embedHidden_1000_lstmHidden_1000_iter_6000.caffemodel')
  parser.add_argument("--classify_weights", type=str, default='snapshots/caption_classifier_embedDrop_75_lstmDrop_75_embedHidden_1000_lstmHidden_1000_iter_6000.caffemodel')

  args = parser.parse_args()

  new_model_weights = transfer_net_weights(args.orig_model, args.orig_model_weights, args.new_model)
  combine_weights = transfer_combine_weights(args.new_model, args.classify_model, new_model_weights, args.classify_weights)
