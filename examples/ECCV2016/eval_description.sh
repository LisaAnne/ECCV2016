
experiment_type=eval_caffe_model 
image_net=/examples/ECCV2016/prototxt/deploy.prototxt

#FINAL LRCN
word_net=/examples/ECCV2016/prototxt/wtd_1000.prototxt
model=gve_models/description
#model=snapshots/description_iter_4000

dataset_name='birds_fg'
split_name='test' #val
#vocab='CUB_vocab'
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
