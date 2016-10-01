#!/bin/sh

experiment_type=eval_class_caffe_model #eval_caffe_model

#featuers lookup
word_net=/examples/ECCV2016/prototxt/wtd_1000.prototxt
model=gve_models/definition
model=snapshots/definition_iter_7347

#lookup_mat=data/lrcn_cc_feat_ccF_feat_ccF_modelSearch_iter_7000_train_gt.p
#lookup_mat=data/lrcn_cc_feat_ccF_feat_ccF_iter_5000_train_gt.p
lookup_mat=/home/lisaanne/caffe-LSTM/examples/finegrained_descriptions/utils_fineGrained/class_embedding/lrcn_cc_feat_ccF_feat_ccF_modelSearch_iter_7000_train_gt_0930.p

size_input_feature=1000
dataset_name='birds_fg'
split_name='test'
#vocab='CUB_vocab_noUNK'
vocab='vocabUNK'

echo $dataset_name
python eval_scripts.py --experiment_type $experiment_type \
                       --model_name $model \
                       --LM_net $word_net \
                       --dataset_name $dataset_name \
                       --split_name $split_name \
                       --vocab $vocab \
                       --prev_word_restriction \
                       --size_input_features $size_input_feature \
                       --lookup_mat $lookup_mat 
#                       --pred
