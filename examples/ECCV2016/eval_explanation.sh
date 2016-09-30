#!/bin/sh

experiment_type=eval_cc_caffe_model #eval_caffe_model

#LSTM
image_net=/examples/finegrained_descriptions/eccv_prototxts/lrcn_cc_lrcn_cc_feat_ccF_feat_ccF_iter_5000_val_gt_feat_feat_deploy.prototxt
#word_net='/examples/finegrained_descriptions/eccv_prototxts/lrcn_cc_lrcn_cc_feat_ccF_feat_ccF_iter_5000_val_gt_feat_feat_wtd.prototxt'
#model='eccv_snapshots/lrcn_cc_lrcn_cc_feat_ccF_feat_ccF_iter_5000_train_gt_feat_feat_modelSearch_iter_7000'

#ind
word_net='examples/finegrained_descriptions/eccv_prototxts/lrcn_reinforce_lstm_classification_sentenceLoss_wbF_ssT_ccT_debug_fromFeatures_wtd.prototxt'

#model=eccv_snapshots/lrcn_reinforce_lstm_classification_sentenceLoss_wbF_ssT_ccT_debug_fromFeatures_modelSearch_swl70_iter_10000

#swl70
model=snapshots/lrcn_reinforce_lstm_classification_sentenceLoss_wbF_ssT_ccT_debug_fromFeatures_train7000_modelSearch_swl80_codeCleanUp2_iter_7000
model=snapshots/lrcn_reinforce_lstm_classification_sentenceLoss_wbF_ssT_ccT_debug_fromFeatures_train7000_modelSearch_swl100_codeCleanUp2_iter_5000
model=snapshots/lrcn_reinforce_lstm_classification_sentenceLoss_wbF_ssT_ccT_debug_fromFeatures_train3000_modelSearch_swl110_codeCleanUp2_iter_3000

size_input_features=2000
label_scale=1
dataset_name='birds_fg'
split_name='val'
vocab='CUB_vocab_noUNK'
lookup_mat=utils_fineGrained/class_embedding/lrcn_cc_feat_ccF_feat_ccF_modelSearch_iter_7000_train_gt.p
precomputed_h5='/yy2/lisaanne/fine_grained/bilinear_features/finegrained/CUB_feature_dict.p'

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
                       --precomputed_h5 $precomputed_h5 \
                       --lookup_mat $lookup_mat 
#                       --pred 

