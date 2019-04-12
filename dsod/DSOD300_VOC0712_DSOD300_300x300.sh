#cd /home/opt/hll/ssd
./build/tools/caffe train \
--solver="examples/project/dsod/models/solver.prototxt" \
--weights='examples/project/dsod/models/pretrained.caffemodel' \
--gpu 0 2>&1 | tee examples/project/dsod/models/DSOD300_VOC0712_DSOD300_300x300.log
