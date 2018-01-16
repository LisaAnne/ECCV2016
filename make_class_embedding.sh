#!/bin/sh

#image input conditioned vector
#image_net=prototxt/deploy.prototxt
#word_net=prototxt/wtd_1000_all.prototxt
##model=snapshots/description_1006
#model=snapshots/description_010517_iter_10000
#
#size_input_feature=1000
#
#dataset_name='birds_fg'
#split_name='train_noCub'
#vocab='vocab'
#
#echo $dataset_name
#python utils/extract_train_val.py --model_name $model \
#                       --LM_net $word_net \
#                       --dataset_name $dataset_name \
#                       --split_name $split_name \
#                       --vocab $vocab \
#                       --image_net $image_net \
#                       --size_input_feature $size_input_feature \
#                       --prev_word_restriction \
#                       --image_input \
#                       --precomputed_h5 data/CUB_feature_010517.p 

image_net=prototxt/deploy.prototxt
word_net=prototxt/lm_wtd.prototxt
model=snapshots/lm_iter_5000

dataset_name='birds_fg'
split_name='train_noCub'
vocab='vocab'

size_input_feature=1000

echo $dataset_name
python utils/extract_train_val.py --model_name $model \
                       --LM_net $word_net \
                       --dataset_name $dataset_name \
                       --split_name $split_name \
                       --vocab $vocab \
                       --image_net $image_net \
                       --size_input_feature $size_input_feature \
                       --prev_word_restriction \
                       --precomputed_h5 data/CUB_feature_010517.p 

