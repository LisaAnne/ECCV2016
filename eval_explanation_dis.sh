#!/bin/bash

experiment_type=eval_caffe_model #eval_caffe_model
image_net=prototxt/deploy.prototxt

#featuers lookup
word_net=prototxt/explanation-dis_wtd.prototxt
#model=gve_models/explanation-dis_1006
model=snapshots/explanation-dis_010817_lw100_iter_ITER

size_input_feature=1000
dataset_name='birds_fg'
split_name='val'
#split_name='val'
vocab='vocab'

iter=(1000 2000 3000 4000 5000 6000 7000 8000 9000 10000)
#iter=(4000)

echo $dataset_name
for i in "${iter[@]}"; do
  model_full=${model//ITER/$i}
  echo model $model_full

  python eval_scripts.py --experiment_type $experiment_type \
                         --model_name $model_full \
                         --image_net $image_net \
                         --LM_net $word_net \
                         --dataset_name $dataset_name \
                         --split_name $split_name \
                         --vocab $vocab \
                         --prev_word_restriction \
                         --precomputed_h5 data/CUB_feature_010517.p
done

