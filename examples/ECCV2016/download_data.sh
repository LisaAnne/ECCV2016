#!/bin/bash

#This will download all data you need to run "Generating Visual Explanation" code.  You will need the coco evaluation toolbox as well.

data_files=( "CUB_feature_dict.p" "CUB_label_dict.p" "cub_missing.txt" "bilinear_preds.p" "cub_0917_5cap.tsv" "train_noCub.txt" "val.txt" "test.txt" "lrcn_cc_feat_ccF_feat_ccF_iter_5000_train_gt.p" )
model_files=( "caption_classifier.caffemodel" "definition.caffemodel" "description_unrolled.caffemodel" "description.caffemodel" "explanation-dis.caffemodel" "explanation-label_unrolled.caffemodel" "explanation-label.caffemodel" "explanation.caffemodel" "caption_classifier_2311.caffemodel")

echo "Downloading data..."

mkdir -p data
cd data
for i in "${data_files[@]}"
do 
  echo "Downloading: " $i
  if [ ! -f $i ];
  then
    wget https://people.eecs.berkeley.edu/~lisa_anne/generating_visual_explanations/data/$i
  fi
done
cd ..

echo "Preprocessing text data..."
python utils/preprocess_captions.py --description_type bird \
                                    --splits data/train_noCub.txt,data/val.txt,data/test.txt

echo "Downloading pretrained models..."

mkdir -p gve_models
cd gve_models
for i in "${model_files[@]}"
do 
  echo "Downloading: " $i
  if [ ! -f $i ];
  then
    wget https://people.eecs.berkeley.edu/~lisa_anne/generating_visual_explanations/gve_models/$i
  fi
done
cd ..

mkdir -p prototxt
mkdir -p snapshots

echo "Done downloading and pre-processing data."
