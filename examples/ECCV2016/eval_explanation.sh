#!/bin/bash

experiment_type=eval_cc_caffe_model #eval_caffe_model
image_net=/examples/ECCV2016/prototxt/deploy.prototxt
word_net=/examples/ECCV2016/prototxt/explanation_wtd.prototxt
size_input_features=2000
label_scale=1
dataset_name='birds_fg'
split_name='test'
vocab='vocab'
model=gve_models/explanation_1006
lookup_mat=data/description_sentence_features.p

echo $dataset_name
  python eval_scripts.py --experiment_type $experiment_type \
                         --model_name $model \
                         --image_net $image_net \
                         --LM_net $word_net \
                         --dataset_name $dataset_name \
                         --split_name $split_name \
                         --vocab $vocab \
                         --size_input_features $size_input_features \
                         --label_scale $label_scale \
                         --prev_word_restriction \
                         --precomputed_h5 data/CUB_feature_dict.p \
                         --lookup_mat $lookup_mat \
                         --pred 

