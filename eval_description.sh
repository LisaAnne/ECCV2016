#/bin/bash

experiment_type=eval_caffe_model 
image_net=prototxt/deploy.prototxt

#FINAL LRCN
word_net=prototxt/wtd_1000.prototxt
model=snapshots/description_010517_iter_3000
#model=gve_models/description_1006

dataset_name='birds_fg'
split_name='test' #val
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
                       --precomputed_h5 data/CUB_feature_010517.p 
