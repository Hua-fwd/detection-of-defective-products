import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.cm as cm
import cv2
import time
import pdb
import heapq
import glob
# %matplotlib inline

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'

# Make sure that caffe is on the python path:
import os
import sys
sys.path.append('./python')

import caffe

# gpu:
caffe.set_device(0)
caffe.set_mode_gpu()

# cpu:
# caffe.set_mode_cpu()

from google.protobuf import text_format
from caffe.proto import caffe_pb2

# load PASCAL VOC labels
labelmap_file = 'examples/project/DenseNet/labelmap_voc.prototxt'
model_def = 'examples/project/DenseNet/model/DenseNet_169.prototxt'
model_weights = 'examples/project/DenseNet/model/ds_169_iter_20000.caffemodel'
file = open(labelmap_file, 'r')
labelmap = caffe_pb2.LabelMap()
text_format.Merge(str(file.read()), labelmap)

def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in xrange(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found == True
    return labelnames

def get_target_labelname(labelmap, target):
    num_labels = len(labelmap.item)
    # labelnames = []
    # if type(labels) is not list:
    #     labels = [labels]
    # for label in labels:
    found = False
    for i in xrange(0, num_labels):
        if target == labelmap.item[i].label:
            found = True
            labelname = labelmap.item[i].display_name
            # labelnames.append(labelmap.item[i].display_name)
            break
    assert found == True
    return labelname

#Load the net in the test phase for inference, and configure input preprocessing.
# model_def = 'examples/07+12+coco/deploy.prototxt'
# model_weights = 'examples/07+12+coco/DSOD300_VOC0712+coco.caffemodel'

# model_def = 'examples/lite_DS-64-256-64-1/deploy.prototxt'
# model_weights = 'examples/lite_DS-64-256-64-1/DSOD300_lite_VOC0712.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
# transformer.set_mean('data', np.array([104,117,123])) # mean pixel
transformer.set_mean('data', np.array([63,63,73])) # mean pixel
# transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
#transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

# DSOD detection

# set net to batch size of 1
image_resize = 224
net.blobs['data'].reshape(1,3,image_resize,image_resize)

# set colors
colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()

#Load an image.
# img = "/home/opt602-1/hll/project/data/data/1_AIF1.bmp"
# img = '/home/opt/hll/mxnet-face-master/detection/dataset/40/40_40_0111.jpg'

def demo(fpath):
    # image = caffe.io.load_image(img)
    img = cv2.imread(fpath)
    image = img
    #image = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    # image = image/255.
    # image = img
    # plt.imshow(image)
    image = cv2.resize(image,(image_resize,image_resize))

    #Run the net and examine the top_k results

    transformed_image = transformer.preprocess('data', image)
    net.blobs['data'].data[...] = transformed_image

    # Forward pass.
    detections = net.forward()['prob']

    # Parse the outputs.


    #image = img
    print detections
    # Get detections with confidence higher than 0.6.
    #top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.0]

    label = np.argmax(detections[0])
    score = detections[0][label]
    label_name = get_target_labelname(labelmap, label)
    display_txt_print = '%s: %.2f'%(label_name, score)
    display_txt = label_name
    print display_txt_print
    
    color = colors[label+10]
    txmin = 30
    tymin = 50
    cv2.putText(image, display_txt_print, (int(txmin),int(tymin)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2) #cv2.FONT_HERSHEY_COMPLEX

    fdst = respath + fpath[fpath.rfind('/'):-3] + '_res.bmp'
    cv2.imwrite(fdst, image)
    # plt.show()
    cv2.imshow('frame', image)
    cv2.waitKey(10)
    return image


imgpath = 'data/data_for_cls/*.bmp'
respath = 'data/data_for_cls/res/'
if not os.path.isdir(respath):
    os.mkdir(respath)
files = glob.glob(imgpath)
for f in files:
    demo(f)

