#/bin/bash

#This will build all the nets you will need

python build_nets.py --net_type caption_classifier --embed_drop 0.75 --lstm_drop 0.75 --embed_dim 1000 --lstm_dim 1000 #These parameters were determined by hyper parameter search

python build_nets.py --net_type definition
python build_nets.py --net_type description
python build_nets.py --net_type explanation-label
python build_nets.py --net_type explanation-dis --weights snapshots/description_HEAD_nv_iter_4000.caffemodel
python build_nets.py --net_type explanation --weights snapshots/explanation-label_nv_iter_8000.caffemodel
python build_nets.py --net_type deploy

