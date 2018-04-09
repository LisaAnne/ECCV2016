#!/usr/bin/env bash

GPU_ID=0
export PYTHONPATH='utils/python_layers/:$PYTHONPATH'

./caffe//build/tools/caffe train -solver prototxt/description_solver.prototxt -gpu 0