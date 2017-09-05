import json
import sys

coco_eval_path = PATH_TO_COCO_EVAL_TOOLS 

caffe_dir = '../../'
sys.path.insert(0, caffe_dir + 'python_layers/')
pycaffe_path = '../../python/'
cub_features = 'data/CUB_feature_dict.p'
bird_anno_path_fg = 'data/descriptions_bird.%s.fg.json'
bird_vocab_path = 'data/'

cache_home = 'generated_sentences/'

def read_json(t_file):
  j_file = open(t_file).read()
  return json.loads(j_file)

def save_json(json_dict, save_name):
  with open(save_name, 'w') as outfile:
    json.dump(json_dict, outfile)  

def open_txt(t_file):
  txt_file = open(t_file).readlines()
  return [t.strip() for t in txt_file]



