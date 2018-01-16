#!/usr/bin/env bash

GPU_ID=0
export PYTHONPATH='utils/python_layers/:$PYTHONPATH'

./caffe//build/tools/caffe train -solver prototxt/caption_classifier_embedDrop_75_lstmDrop_75_embedHidden_1000_lstmHidden_1000_solver.prototxt -gpu 0