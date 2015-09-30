#!/usr/bin/env python
import os
import numpy as np
import copy
from dataset import imagenet_vdet_classes
from proposal import vid_proposals
from ..utils.protocol import empty_det_from_box, score_proto, bbox_hash, det_score
from ..utils.common import imread
from ..utils.log import logging
from ..utils.cython_nms import vid_nms

def det_vid_with_box(vid_proto, box_proto, det_fun, net,
                    class_names=imagenet_vdet_classes):
    assert vid_proto['video'] == box_proto['video']
    root = vid_proto['root_path']
    det_proto = empty_det_from_box(box_proto)
    for frame in vid_proto['frames']:
        frame_id, path = frame['frame'], frame['path']
        det_cur_frame = \
            [i for i in det_proto['detections'] if i['frame'] == frame_id]
        if len(det_cur_frame) > 0:
            logging.info("Detecting in frame {}, {} boxes...".format(
                frame_id, len(det_cur_frame)))
            img = imread(os.path.join(root, path))
            boxes = [det['bbox'] for det in det_cur_frame]
            det_scores = det_fun(img, boxes, net)
            for det, scores in zip(det_cur_frame, det_scores):
                det['scores'] = score_proto(class_names, scores)
    return det_proto


def det_vid_without_box(vid_proto, det_fun, net,
                    class_names=imagenet_vdet_classes):
    logging.info("Generating proposals...")
    box_proto = vid_proposals(vid_proto)
    det_proto = det_vid_with_box(vid_proto, box_proto, det_fun, net,
                                 class_names)
    return det_proto


def det_vid_score(vid_proto, det_fun, net, box_proto=None,
                    class_names=imagenet_vdet_classes):
    if box_proto:
        return det_vid_with_box(vid_proto, box_proto, det_fun, net, class_names)
    else:
        return det_vid_without_box(vid_proto, det_fun, net, class_names)


def apply_vid_nms(det_proto, class_index, thres=0.3):
    logging.info('Apply NMS on video: {}'.format(det_proto['video']))
    new_det = {}
    new_det['video'] = det_proto['video']
    boxes = np.asarray([[det['frame'],]+det['bbox']+[det_score(det, class_index),]
             for det in det_proto['detections']], dtype='float32')
    keep = vid_nms(boxes, thresh=0.3)
    new_det['detections'] = copy.copy([det_proto['detections'][i] for i in keep])
    logging.info("{} / {} windows kept.".format(len(new_det['detections']),
                                         len(det_proto['detections'])))
    return new_det


