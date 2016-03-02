#!/usr/bin/env python
import json
import numpy as np
import cv2
import os
from ..utils.common import im_transform, img_crop, rcnn_img_crop, Pool, timeout
from ..utils.log import logging
from ..utils.protocol import proto_load, proto_dump
from ..utils.cython_nms import nms
import sys

def simple_crop(image, bbox):
    bbox = np.asarray(bbox)
    bbox -= 1
    bbox[0] = max(0, bbox[0])
    bbox[1] = max(0, bbox[1])
    bbox[2] = min(image.shape[1] - 1, bbox[2])
    bbox[3] = min(image.shape[0] - 1, bbox[3])
    return image[bbox[1]:bbox[3]+1, bbox[0]:bbox[2]+1, :]


def fast_rcnn_det(img, boxes, net):
    from fast_rcnn.test import im_detect
    # suppress caffe logs
    try:
        orig_loglevel = os.environ['GLOG_minloglevel']
    except KeyError:
        orig_loglevel = '0'
    os.environ['GLOG_minloglevel'] = '2'

    scores, reg_boxes = im_detect(net, img, np.array(boxes))
    os.environ['GLOG_minloglevel'] = orig_loglevel
    return scores


@timeout(30)
def googlenet_det(img, bbox, net):
    # suppress caffe logs
    try:
        orig_loglevel = os.environ['GLOG_minloglevel']
    except KeyError:
        orig_loglevel = '0'
    os.environ['GLOG_minloglevel'] = '2'

    size = 224
    mean_values = [103.939, 116.779, 123.68]
    patch = im_transform(simple_crop(img, bbox), size, 1., mean_values)
    net.blobs['data'].data[...] = patch
    net.forward()
    det_scores = np.copy(net.blobs['cls_score'].data[0])

    os.environ['GLOG_minloglevel'] = orig_loglevel
    return det_scores.tolist()


def googlenet_rcnn(img, boxes, net):
    return googlenet_features(img, boxes, net, 'cls_score')


class RCNNProcesser(object):
    """docstring for RCNNProcesser"""
    def __init__(self, img, crop_mode, crop_size, pad_size, mean_values):
        self.img = img
        self.crop_mode = crop_mode
        self.crop_size = crop_size
        self.pad_size = pad_size
        self.mean_values = mean_values

    def __call__(self, input):
        return im_transform(
            rcnn_img_crop(self.img, input, self.crop_mode,
                          self.crop_size, self.pad_size,
                          self.mean_values))

def googlenet_features(img, boxes, net, blob_name):
    # suppress caffe logs
    try:
        orig_loglevel = os.environ['GLOG_minloglevel']
    except KeyError:
        orig_loglevel = '0'
    os.environ['GLOG_minloglevel'] = '2'

    size = 224
    mean_values = np.asarray([103.939, 116.779, 123.68])
    batch_size = 128
    slice_points = np.arange(0, len(boxes), batch_size)[1:]
    features = None
    processer = RCNNProcesser(img, 'warp', size, 16, mean_values)

    for batch_boxes in np.split(np.asarray(boxes), slice_points):
        patches = np.asarray(map(processer, batch_boxes))
        net.blobs['data'].reshape(*(patches.shape))
        net.blobs['data'].data[...] = patches
        net.forward()
        cur_feat = net.blobs[blob_name].data
        if features is None:
            features = np.copy(cur_feat)
        else:
            try:
                features = np.r_[features, cur_feat]
            except ValueError, e:
                logging.error("Unable to concatenate features.")
                print features.shape, cur_feat.shape
                raise e
    os.environ['GLOG_minloglevel'] = orig_loglevel
    return np.asarray(features)


def svm_scores(features, svm_model):
    if features.ndim == 4:
        features = np.squeeze(features, axis=(2,3))
    features = np.asarray(features) * (20. / svm_model['feat_norm_mean'])
    scores = np.dot(features, svm_model['W']) + svm_model['B']
    return scores


def apply_image_nms(boxes, scores, thres=0.3):
    box_score = np.asarray(np.r_['-1', boxes, np.reshape(scores, (-1,1))],
                           dtype='float32')
    logging.info("Applying nms to image.")
    keep = nms(box_score, thres)
    logging.info("{} / {} boxes kept.".format(len(keep), len(boxes)))
    return keep
