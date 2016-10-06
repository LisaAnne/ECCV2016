#!/bin/sh

experiment_type=eval_caffe_model #eval_caffe_model
image_net=/examples/ECCV2016/prototxt/deploy.prototxt

#featuers lookup
word_net=/examples/ECCV2016/prototxt/explanation-dis_wtd.prototxt
model=gve_models/explanation-dis_1006

size_input_feature=1000
dataset_name='birds_fg'
split_name='test'
#split_name='val'
vocab='vocab'

echo $dataset_name
python eval_scripts.py --experiment_type $experiment_type \
                       --model_name $model \
                       --image_net $image_net \
                       --LM_net $word_net \
                       --dataset_name $dataset_name \
                       --split_name $split_name \
                       --vocab $vocab \
                       --prev_word_restriction \
                       --precomputed_h5 data/CUB_feature_dict.p 

