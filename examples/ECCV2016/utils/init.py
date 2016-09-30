import json
import sys

cub_features = 'data/CUB_feature_dict.p'

caffe_dir = '/home/lisaanne/caffe-LSTM/'
coco_data_dir = '/home/lisaanne/caffe-LSTM/data/coco/'
sys.path.insert(0, caffe_dir + 'python_layers/')
pycaffe_path = '/home/lisaanne/caffe-LSTM/python/'

bird_dataset_path = '/yy2/lisaanne/fine_grained/CUB_200_2011/'
bird_image_path = bird_dataset_path + '/images/'
bird_anno_path = '/home/lisaanne/caffe-LSTM/examples/finegrained_descriptions/utils_fineGrained/descriptions/descriptions_bird.%sclasses.zs.json' 
bird_anno_path_fg = '/home/lisaanne/caffe-LSTM/examples/finegrained_descriptions/utils_fineGrained/descriptions/descriptions_bird.%s.fg_%s.json'
bird_vocab_path = '/home/lisaanne/caffe-LSTM/examples/finegrained_descriptions/utils_fineGrained/vocab/' 
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



