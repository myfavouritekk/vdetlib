#!/usr/bin/env python

import os
from ..utils.common import read_list

datasets_root = os.path.join(os.path.dirname(__file__), '../misc/')
imagenet_vdet_classes = read_list(os.path.join(datasets_root,
     'imagenet_vdet_classes.txt'))
imagenet_vdet_class_idx = dict(zip(imagenet_vdet_classes,
                           xrange(len(imagenet_vdet_classes))))

imagenet_det_200_classes = read_list(os.path.join(datasets_root,
     'imagenet_det_200_classes.txt'))
imagenet_det_200_class_idx = dict(zip(imagenet_det_200_classes,
                          xrange(len(imagenet_det_200_classes))))

index_vdet_to_det = dict([(imagenet_vdet_classes.index(vdet_class),
                           imagenet_det_200_classes.index(vdet_class)) \
                        for vdet_class in imagenet_vdet_classes])

