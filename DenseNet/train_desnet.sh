#!/usr/bin/env sh
set -e

./build/tools/caffe train \
    --solver=examples/project/DenseNet/model/solver.prototxt \
    --weights=examples/project/DenseNet/model/DenseNet_169.caffemodel  \
    --gpu 1 $@ 2>&1 | tee examples/project/DenseNet/model/log.txt

