#!/usr/bin/env bash

GPU_ID=0
WEIGHTS=snapshots/description_1006indLSTM_caption_classifier_1006.caffemodel

export PYTHONPATH='utils/python_layers/:$PYTHONPATH'

./caffe//build/tools/caffe train -solver prototxt/explanation-dis_solver.prototxt -weights snapshots/description_1006indLSTM_caption_classifier_1006.caffemodel -gpu 0