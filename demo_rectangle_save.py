#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
from scipy import misc
import sys
from random import shuffle
CLASSES = ('__background__',
           'build', 'nobuild')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'Alibaba_ZF_faster_rcnn_final_3.caffemodel')}

 
fp_change = open("change_rect.txt","w+")

 
def save_rect_info(fp,dets,img_name,thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        
        #box parameters
        x = bbox[0]
        y = bbox[1]
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        s = score
        result = "%s %d %d %d %d %.2f\n" % (img_name,x,y,w,h,s)
        fp.write(result);
                      

def detections(net,im,tag,NMS_THRESH,reverse=False):
    scores, boxes = im_detect(net, im)
    # Visualize detections for each class
    if not reverse:
        cls_ind = 1 if tag == '2017' else 2# because we skipped background
        cls = 'build' if tag == '2017' else 'nobuild'
    else:
        cls_ind = 2 if tag == '2017' else 1# because we skipped background
        cls = 'nobuild' if tag == '2017' else 'build'

    cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
    cls_scores = scores[:, cls_ind]
    dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
    keep = nms(dets, NMS_THRESH)
    dets = dets[keep, :]
    return dets,cls
def dist(target,probs):
    im=np.zeros((256,256))
    xmin,ymin,xmax,ymax = target.astype(np.int)
    w=xmax-xmin
    h=ymax-ymin
    im[ymin:ymin+h,xmin:xmin+w]=1


    imo=np.zeros((256,256))
    for prob in probs:
        xmin,ymin,xmax,ymax = prob.astype(np.int)
        w=xmax-xmin
        h=ymax-ymin
        imo[ymin:ymin+h,xmin:xmin+w]=1
    
    return np.sum(im*imo)/np.sum(im)
def vis_detections(im, class_name,dets,dets_other,image_name,thresh=0.0):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    inds_other = dets_other[:, -1] >= thresh
    if len(inds) == 0 or len(inds_other) == 0:
        return

    #im = im[:, :, (2, 1, 0)]
    #fig, ax = plt.subplots(figsize=(12, 12))
    #ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        Ja = dist(bbox,dets_other[inds_other,:4])
        if Ja < thresh:
            continue
                   
        xmin,ymin,xmax,ymax = bbox.astype(np.int)
        w=xmax-xmin
        h=ymax-ymin
        im[ymin:ymin+h,xmin:xmin+w]=1
        
        x=xmin
        y=ymin
        result = "%s %d %d %d %d\n" % (image_name,x,y,w,h)
        fp_change.write(result)
        

def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    name = ['2017','2015']
    CONF_THRESH = 0.55
    NMS_THRESH = 0.3
    res=np.zeros((256,256)) # put change in res
    f2015 = open('2015rect.txt','a+')
    f2017 = open('2017rect.txt','a+')
    for ikeep,tag in enumerate(name):
        if ikeep:
            break
        im_file = os.path.join(tag, image_name)
        im = cv2.imread(im_file) #2o17 is need build
        im_file = os.path.join(name[(ikeep+1)%2], image_name)
        im_other = cv2.imread(im_file)# 2015 is need nobuild
        # Detect all object classes and regress object bounds
        timer = Timer()
        timer.tic()  #done two choice is catch change 
        
        if tag == '2017':
            fp1 = f2017
            fp2 = f2015
        else:
            fp1 = f2015
            fp2 = f2017
        
        dets,cls = detections(net,im,tag,NMS_THRESH)
        save_rect_info(fp1,dets,image_name,thresh=CONF_THRESH)
        dets_other,cls2 = detections(net,im_other,tag,NMS_THRESH,True)
        save_rect_info(fp2,dets_other,image_name,thresh=CONF_THRESH)
        timer.toc()
        print ('Detection took {:.3f}s for '
            '{:d} object proposals').format(timer.total_time, dets.shape[0])

        vis_detections(res,cls, dets,dets_other, image_name,thresh=CONF_THRESH)
    misc.imsave('output/stride32_change_new/'+image_name,res)
    f2015.close()
    f2017.close()
    fp_change.close()

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
                            'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
                              NETS[args.demo_net][1])

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 256 * np.ones((256, 256, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)
    
    im_names = [line for line in open('test.txt','r').read().splitlines()]
    #shuffle(im_names)
    #im_names = im_names[:12]
    for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for data/demo/{}'.format(im_name)
        demo(net, im_name)

    plt.show()
