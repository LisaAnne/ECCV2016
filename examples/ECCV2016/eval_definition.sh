#!/bin/sh

experiment_type=eval_class_caffe_model #eval_caffe_model

#featuers lookup
word_net=/examples/finegrained_descriptions/eccv_prototxts/class_only_lrcn_vector_wtd.prototxt
model=eccv_snapshots/class_only_lrcn_vector_modelSearch_iter_7000

#word_net=examples/finegrained_descriptions/prototxt/lrcn_reinforce_lstm_classification_sentenceLoss_wbF_ssT_ccT_debug_labelOnly_fromFeatures_wtd.prototxt
#model=snapshots/class_only_lrcn_vector_modelSearch_iter_7000indLSTM_caption_classifier_embedDrop_75_lstmDrop_75_embedHidden_1000_lstmHidden_1000_iter_6000
#model=snapshots/lrcn_reinforce_lstm_classification_sentenceLoss_wbF_ssT_ccT_debug_labelOnly_fromFeatures_lamda70_iter_10000
#model=snapshots/lrcn_reinforce_lstm_classification_sentenceLoss_wbF_ssT_ccT_debug_labelOnly_fromFeatures_lamda80_iter_10000

lookup_mat=utils_fineGrained/class_embedding/lrcn_cc_feat_ccF_feat_ccF_modelSearch_iter_7000_train_gt.p

size_input_feature=1000
dataset_name='birds_fg'
split_name='val'
#split_name='val'
vocab='CUB_vocab_noUNK'

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
