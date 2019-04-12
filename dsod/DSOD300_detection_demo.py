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
labelmap_file = 'examples/project/dsod/labelmap_voc.prototxt'
file = open(labelmap_file, 'r')
labelmap = caffe_pb2.LabelMap()
text_format.Merge(str(file.read()), labelmap)

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
model_def = 'examples/project/dsod/model/deploy.prototxt'
model_weights = 'examples/project/dsod/model/DSOD300_iter_12000.caffemodel'

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
image_resize = 300
net.blobs['data'].reshape(1,3,image_resize,image_resize)

# set colors
colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()

def choose_det(dets):
    print "Parse the dets"

    # Parse the outputs.
    # det_label = dets[:,1]
    # det_conf = dets[:,2]
    # det_xmin = dets[:,3]
    # det_ymin = dets[:,4]
    # det_xmax = dets[:,5]
    # det_ymax = dets[:,6]
    print dets
    if dets.shape[0] == 0:
        return dets
    det_xmax = dets[:,5]
    threshold = 0.15
    right_indices = [i for i, xmax in enumerate(det_xmax) if xmax >= threshold]
    dets = dets[right_indices,:]

    print dets
    if dets.shape[0] == 0:
        return dets
    # second choose max conf
    det_conf = dets[:,2]
    #target = np.argmax(det_conf)
    list_conf = det_conf.tolist()
    target = map(list_conf.index, heapq.nlargest(2, list_conf))

    return dets[target,:]

def demo(fpath):
    # image = caffe.io.load_image(img)
    img = cv2.imread(fpath)
    image = img
    #image = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    image = cv2.resize(image,(300,300))

    #Run the net and examine the top_k results

    transformed_image = transformer.preprocess('data', image)
    net.blobs['data'].data[...] = transformed_image

    # Forward pass.
    detections = net.forward()['detection_out']

    # Parse the outputs.
    det_label = detections[0,0,:,1]
    det_conf = detections[0,0,:,2]
    det_xmin = detections[0,0,:,3]
    det_ymin = detections[0,0,:,4]
    det_xmax = detections[0,0,:,5]
    det_ymax = detections[0,0,:,6]

    #image = img
    print det_label
    # Get detections with confidence higher than 0.6.
    top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.0]
    top_detecs = detections[0,0,top_indices,:]
    dets = choose_det(top_detecs)
    print type(dets)
    print dets
    if dets.shape[0] == 0:
        cv2.imshow('frame', image)
        cv2.waitKey(1000)
        return image
    for det in dets:

        label = int(det[1])
        score = det[2]
        label_name = get_target_labelname(labelmap, label)
        display_txt_print = '%s: %.2f'%(label_name, score)
        display_txt = label_name
        print display_txt_print
        
        #pdb.set_trace()
        # draw rectangle
        top_xmin = det[3]
        top_ymin = det[4]
        top_xmax = det[5]
        top_ymax = det[6]
        color = colors[label+10]
        xmin = int(round(top_xmin * image.shape[1]))
        ymin = int(round(top_ymin * image.shape[0]))
        xmax = int(round(top_xmax * image.shape[1]))
        ymax = int(round(top_ymax * image.shape[0]))
        cv2.rectangle(image,(int(xmin),int(ymin)),(int(xmax),int(ymax)),(0, 255, 0),1)
        txmin = max(xmin-50,0)
        tymin = max(ymin-10,10)
        cv2.putText(image, display_txt_print, (int(txmin),int(tymin)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1) #cv2.FONT_HERSHEY_COMPLEX
        print xmin,ymin

    # plt.show()
    fdst = respath + fpath[fpath.rfind('/'):-3] + '_res.bmp'
    cv2.imwrite(fdst, image)
    cv2.imshow('frame', image)
    cv2.waitKey(10)
    return image


imgpath = 'data/data/*.bmp'
respath = 'data/data/res/'
if not os.path.isdir(respath):
    os.mkdir(respath)
files = glob.glob(imgpath)
for f in files:
    demo(f)


