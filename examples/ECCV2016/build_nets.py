import argparse
import sys
sys.path.append('utils/')
try:
  from init import * 
except:
  print "Please update utils/init.py to reflect your environment.  Copy utils/init.example.py to utils/init.py and update to match the paths on your machine"
from build_net import lrcn, reinforce
from build_net import caffe_net as cn 
from transfer_to_ind_LSTM import transfer_net_weights, transfer_combine_weights
import pdb

test_set = 'val'

def make_lrcn_param_str(label_options = {'label_format': 'number'}, data_layer='pairedCaptionData'):
  
  param_str_train = {}
  param_str_test = {}
  param_str_train['caption_json'] = bird_anno_path_fg %('train_noCub')
  param_str_test['caption_json'] = bird_anno_path_fg %(test_set)
  param_str_train['vocabulary'] = 'data/vocab.txt' 
  param_str_test['vocabulary'] = 'data/vocab.txt' 

  output_keys = [('text_data_key', 'input_sentence'), ('text_label_key', 'target_sentence'), ('text_marker_key', 'cont_sentence'), ('image_data_key', 'image_data'), ('data_label', 'data_label'), ('data_label_feat', 'data_label_feat')]
  top_names = ['input_sentence', 'target_sentence', 'cont_sentence', 'image_data', 'data_label', 'data_label_feat']

  for key, value in output_keys:
    param_str_train[key] = value
    param_str_test[key] = value
 
  for key in label_options:
    param_str_train[key] = label_options[key]
    param_str_test[key] = label_options[key]

  param_str_train['top_names'] = top_names
  param_str_test['top_names'] = top_names

  return param_str_train, param_str_test

def make_lrcn_class_param_str(label_options = {'label_format': 'number', 'sentence_supervision': 'all', 'label_stream_size': 1}):
  
  param_str_train = {}
  param_str_test = {}
  param_str_train['caption_json'] = bird_anno_path_fg %('train_noCub')
  param_str_test['caption_json'] = bird_anno_path_fg %('val')
  param_str_train['vocabulary'] = 'data/vocab.txt' 
  param_str_test['vocabulary'] = 'data/vocab.txt' 
 
  output_keys = [('text_data_key', 'input_sentence'), ('text_label_key', 'target_sentence'), ('text_marker_key', 'cont_sentence'), ('data_label', 'data_label')]
  top_names = ['input_sentence', 'target_sentence', 'cont_sentence', 'data_label']


  for key, value in output_keys:
    param_str_train[key] = value
    param_str_test[key] = value
 
  for key in label_options:
    param_str_train[key] = label_options[key]
    param_str_test[key] = label_options[key]

  param_str_train['top_names'] = top_names
  param_str_test['top_names'] = top_names

  return param_str_train, param_str_test

def build_sentence_generation_deploy():
  data_inputs = {}
  data_inputs['param_str'] ={'vocabulary': 'data/vocab.txt'}
  model_train = lrcn.lrcn(data_inputs, lstm_dim=1000, embed_dim=1000, class_conditional=True, image_conditional=True, class_size=200, image_dim=8192)
  model_train.make_sentence_generation_deploy() 

def build_sentence_generation_model(model_id, class_conditional, image_conditional, solver_args={'max_iter': 15000, 'stepsize': 2000, 'snapshot': 1000}):

  layer = 'extractGVEFeatures' 

  save_file_name_base = 'prototxt/%s' %(model_id)
  save_file_train = '%s_%s.prototxt' %(save_file_name_base, 'train')
  save_file_test_on_train = '%s_%s.prototxt' %(save_file_name_base, 'test_on_train')
  save_file_test_on_test = '%s_%s.prototxt' %(save_file_name_base, 'test_on_test')
  save_file_deploy = '%s_%s.prototxt' %(save_file_name_base,'deploy')
  save_file_solver = '%s_%s.prototxt' %(save_file_name_base,'solver')
  save_bash = '%s_%s.sh' %('train', model_id)
 
  data_inputs_train = {}
  data_inputs_train['module'] = 'data_layers' 
  data_inputs_train['layer'] = layer 
  data_inputs_test = {}
  data_inputs_test['module'] = 'data_layers' 
  data_inputs_test['layer'] = layer

  data_inputs_train = data_inputs_train
  data_inputs_test = data_inputs_test
  label_options = {'vector_file': 'data/description_HEAD_nv_iter_4000_train_noCub_gt_0930.p'}
  param_str_train, param_str_test = make_lrcn_param_str(label_options=label_options) 
 
  data_inputs_train['param_str'] = param_str_train
  data_inputs_test['param_str'] = param_str_test

  model_train = lrcn.lrcn(data_inputs_train, lstm_dim=1000, embed_dim=1000, class_conditional=class_conditional, image_conditional=image_conditional, class_size=200, image_dim=8192)
  model_train.make_sentence_generation_net(save_file_train, accuracy=False, loss=True) 
  cn.make_solver(save_file_solver, [save_file_train], [], **solver_args)
  cn.make_bash_script(save_bash, save_file_solver)

def caption_classifier(embed_dim, lstm_dim, embed_drop, lstm_drop):

  save_file_name_base = 'caption_classifier_embedDrop_%s_lstmDrop_%s_embedHidden_%s_lstmHidden_%s' %(int(embed_drop*100), int(lstm_drop*100), int(embed_dim), int(lstm_dim))
  save_file_train = 'prototxt/%s_%s.prototxt' %(save_file_name_base, 'train')
  save_file_test_on_train = 'prototxt/%s_%s.prototxt' %(save_file_name_base, 'test_on_train')
  save_file_test_on_test = 'prototxt/%s_%s.prototxt' %(save_file_name_base, 'test_on_test')
  save_file_deploy = 'prototxt/%s_deploy.prototxt' %save_file_name_base
  save_file_solver = 'prototxt/%s_%s.prototxt' %(save_file_name_base, 'solver')
  save_bash = '%s_%s.sh' %('train', save_file_name_base)

  data_inputs_train = {}
  data_inputs_train['module'] = 'data_layers' 
  data_inputs_train['layer'] = 'CaptionToLabel'
  data_inputs_test = {}
  data_inputs_test['module'] = 'data_layers' 
  data_inputs_test['layer'] = 'CaptionToLabel'

  label_options = {'sentence_supervision': 'last', 'label_stream_size': 20}
  param_str_train, param_str_test = make_lrcn_class_param_str(label_options=label_options) 
  
  data_inputs_train['param_str'] = param_str_train
  data_inputs_test['param_str'] = param_str_test

  model_train = lrcn.lrcn(data_inputs_train, lstm_dim=lstm_dim, embed_dim=embed_dim)
  model_train.caption_classifier(save_file_train, accuracy=False, loss=True, embed_drop=embed_drop, lstm_drop=lstm_drop)
  model_train.caption_classifier(save_file_deploy, accuracy=False, loss=False, deploy=True, embed_drop=embed_drop, lstm_drop=lstm_drop)

  model_test_on_train = lrcn.lrcn(data_inputs_train, lstm_dim=lstm_dim, embed_dim=embed_dim)
  model_test_on_train.caption_classifier(save_file_test_on_train, accuracy=True, loss=False)

  model_test_on_test = lrcn.lrcn(data_inputs_test, lstm_dim=lstm_dim, embed_dim=embed_dim)
  model_test_on_test.caption_classifier(save_file_test_on_test, accuracy=True, loss=False)

  cn.make_solver(save_file_solver, [save_file_train], [save_file_test_on_train, save_file_test_on_test], **{'base_lr': 0.1, 'stepsize': 2000, 'max_iter': 6000})
  cn.make_bash_script(save_bash, save_file_solver)

def sentence_generation_reinforce(save_file_name, weights=None, orig_proto=None, classify_model=None, classify_weights=None, RL_loss='lstm_classification', class_conditional=True, lw=20):


  save_file_name_base = 'prototxt/%s' %save_file_name
  save_file_train = '%s_%s.prototxt' %(save_file_name_base, 'train')
  save_file_test_on_train = '%s_%s.prototxt' %(save_file_name_base, 'test_on_train')
  save_file_test_on_test = '%s_%s.prototxt' %(save_file_name_base, 'test_on_test')
  save_file_deploy = '%s_%%s.prototxt' %save_file_name_base
  save_file_solver = '%s_%s.prototxt' %(save_file_name_base, 'solver')
  save_bash = '%s_%s.sh' %('train', save_file_name)
  
  label_options = {'vector_file': 'data/description_HEAD_nv_iter_4000_train_noCub_gt_0930.p'}

  data_layer = 'extractGVEFeatures'
  data_inputs_train = {}
  data_inputs_train['module'] = 'data_layers' 
  data_inputs_train['layer'] = data_layer 
  data_inputs_test = {}
  data_inputs_test['module'] = 'data_layers' 
  data_inputs_test['layer'] = data_layer

  param_str_train, param_str_test = make_lrcn_param_str(label_options=label_options, data_layer=data_layer) 

  data_inputs_train['param_str'] = param_str_train
  data_inputs_test['param_str'] = param_str_test

  model_train = reinforce.reinforce(data_inputs_train, cc=class_conditional, baseline=False, separate_sents=True)
  model_train.lrcn_reinforce(save_name=save_file_train, RL_loss=RL_loss, lw=lw)
 
  model_lm_deploy = reinforce.reinforce(data_inputs_test, cc=class_conditional, T=1)
  model_lm_deploy.lrcn_reinforce_wtd_deploy(save_name=save_file_deploy %'wtd')

  cn.make_solver(save_file_solver, [save_file_train], [], 
                  **{'base_lr': 0.001, 'stepsize': 2000, 'max_iter': 10000, 'snapshot': 1000})

  if weights:
    ind_model_weights = transfer_net_weights(orig_proto, weights, save_file_train) 
    save_file_train = transfer_combine_weights(save_file_train, classify_model, ind_model_weights, classify_weights) 
  
  if weights: 
    cn.make_bash_script(save_bash, save_file_solver, weights=save_file_train)
  else:
    cn.make_bash_script(save_bash, save_file_solver)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--net_type",type=str)
  parser.add_argument("--class_label",type=str, default=None)
  parser.add_argument("--embed_drop",type=float, default=0)
  parser.add_argument("--lstm_drop",type=float, default=0)
  parser.add_argument("--embed_dim",type=float, default=1000)
  parser.add_argument("--lstm_dim",type=float, default=1000)
  parser.add_argument("--weights",type=str, default=None)
  parser.add_argument("--classify_model",type=str, default='prototxt/caption_classifier_embedDrop_75_lstmDrop_75_embedHidden_1000_lstmHidden_1000_train.prototxt')
  parser.add_argument("--classify_weights",type=str, default='snapshots/caption_classifier_embedDrop_75_lstmDrop_75_embedHidden_1000_lstmHidden_1000_iter_6000.caffemodel')

  args = parser.parse_args()
  
  if args.net_type == 'definition':  
    build_sentence_generation_model('definition', True, False)
  elif args.net_type == 'description':  
    build_sentence_generation_model('description', False, True)
  elif args.net_type == 'explanation-label':
    build_sentence_generation_model('explanation-label', True, True)
  elif args.net_type == 'deploy':
    build_sentence_generation_deploy() 
  elif args.net_type == 'explanation-dis':
    sentence_generation_reinforce('explanation-dis', orig_proto='prototxt/description_train.prototxt', classify_model=args.classify_model, classify_weights=args.classify_weights,  weights=args.weights, class_conditional=False, lw=80)  #Loss weight parameter chosen by parameter search
  elif args.net_type == 'explanation':
    sentence_generation_reinforce('explanation', orig_proto='prototxt/explanation-label_train.prototxt', classify_model=args.classify_model, classify_weights=args.classify_weights,  weights=args.weights, class_conditional=True, lw=110)  #Loss weight parameter chocen by parameter search
  elif args.net_type == 'caption_classifier':
    caption_classifier(int(args.embed_dim), 
                       int(args.lstm_dim), 
                       float(args.embed_drop), 
                       float(args.lstm_drop))
  else: 
    raise Exception("Did not select valid experiment type.") 
