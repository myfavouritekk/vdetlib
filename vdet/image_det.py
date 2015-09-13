#!/usr/bin/env python
import json
import numpy as np
import cv2
import os
from ..utils.common import im_transform
from ..utils.protocol import proto_load, proto_dump


def crop(image, bbox):
    bbox -= 1
    return image[bbox[1]:bbox[3]+1, bbox[0]:bbox[2]+1]


def googlenet_det(img, bbox, net):
    size = 224
    mean_values = [103.939, 116.779, 123.68]
    patch = im_transform(crop(img, bbox), size, 1., mean_values)
    net.blobs['data'].data[...] = patch
    net.forward()
    det_scores = np.copy(net.blobs['cls_score'].data[0])
    return det_scores.tolist()

