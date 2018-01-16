#!/bin/bash

experiment_type=eval_cc_caffe_model #eval_caffe_model
image_net=prototxt/deploy.prototxt
word_net=prototxt/explanation_wtd.prototxt
size_input_features=2000
label_scale=1
dataset_name='birds_fg'
split_name='test'
vocab='vocab'
model=gve_models/explanation_1006
model=snapshots/explanation_010615_lw80_iter_ITER
#lookup_mat=data/description_sentence_features.p
lookup_mat=data/description_010517_iter_10000_train_noCub_gt_0930.p

#iter=(1000 2000 3000 4000 5000 6000 7000 8000 9000 10000)
iter=(6000)

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
                         --size_input_features $size_input_features \
                         --label_scale $label_scale \
                         --prev_word_restriction \
                         --precomputed_h5 data/CUB_feature_010517.p \
                         --lookup_mat $lookup_mat \
                         #--pred 
                         #--precomputed_h5 data/CUB_feature_dict.p \
done
