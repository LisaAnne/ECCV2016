#!/bin/bash

#This will download all data you need to run "Generating Visual Explanation" code.  You will need the coco evaluation toolbox as well.

data_files=( "CUB_feature_dict.p" "CUB_label_dict.p" "bilinear_preds.p" "cub_0917_5cap.tsv" "train_noCub.txt" "val.txt" "test.txt" "description_sentence_features.p" "CUB_vocab_noUNK.txt")
model_files=( "caption_classifier_1006.caffemodel" "definition_1006.caffemodel"  "description_1006.caffemodel" "explanation-dis_1006.caffemodel"  "explanation-label_1006.caffemodel" "explanation_1006.caffemodel" )
cider_scores=( "cider_score_dict_definition.p" "cider_score_dict_description.p" "cider_score_dict_explanation-dis.p" "cider_score_dict_explanation-label.p" "cider_score_dict_explanation.p" )

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

echo "Downloading cider scores..."
mkdir -p cider_scores 
cd cider_scores
for i in "${cider_scores[@]}"
do 
  echo "Downloading: " $i
  if [ ! -f $i ];
  then
    wget https://people.eecs.berkeley.edu/~lisa_anne/generating_visual_explanations/cider_scores/$i
  fi
done
cd ..


mkdir -p prototxt
mkdir -p snapshots

echo "Done downloading and pre-processing data."
