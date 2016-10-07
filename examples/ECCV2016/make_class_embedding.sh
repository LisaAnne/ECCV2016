#!/bin/sh

image_net=/examples/ECCV2016/prototxt/deploy.prototxt
word_net=/examples/finegrained_descriptions/eccv_prototxts//lrcn_cc_feat_ccF_feat_ccF_wtd_all.prototxt
model=snapshots/description_HEAD_nv_iter_4000

size_input_feature=1000

dataset_name='birds_fg'
split_name='train_noCub'
vocab='vocab'

echo $dataset_name
python extract_train_val.py --model_name $model \
                       --LM_net $word_net \
                       --dataset_name $dataset_name \
                       --split_name $split_name \
                       --vocab $vocab \
                       --image_net $image_net \
                       --size_input_feature $size_input_feature \
                       --prev_word_restriction \
                       --precomputed_h5 data/CUB_feature_dict.p 
