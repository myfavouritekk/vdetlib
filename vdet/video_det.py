#!/usr/bin/env python
import os
import numpy as np
import copy
from dataset import imagenet_vdet_classes
from proposal import get_windows
from ..utils.protocol import empty_det_from_box, score_proto, bbox_hash, det_score
from ..utils.common import imread
from ..utils.cython_nms import vid_nms

def det_vid_with_box(vid_proto, box_proto, det_fun, net,
                    class_names=imagenet_vdet_classes):
    assert vid_proto['video'] == box_proto['video']
    root = vid_proto['root_path']
    det_proto = empty_det_from_box(box_proto)
    for frame in vid_proto['frames']:
        frame_id, path = frame['frame'], frame['path']
        img = imread(os.path.join(root, path))
        det_cur_frame = \
            [i for i in det_proto['detections'] if i['frame'] == frame_id]
        print "Detecting in frame {}, {} boxes...".format(
            frame_id, len(det_cur_frame))
        for det in det_cur_frame:
            scores = det_fun(img, det['bbox'], net)
            det['scores'] = score_proto(class_names, scores)
    return det_proto


def det_vid_without_box(vid_proto, det_fun, net,
                    class_names=imagenet_vdet_classes):
    root = vid_proto['root_path']
    det_proto = {}
    det_proto['video'] = vid_proto['video']
    detections = []
    frame_names = [os.path.join(root, frame['path']) for frame in vid_proto['frames']]
    print "Generating proposals..."
    all_boxes = get_windows(frame_names)
    print "Detecting..."
    for frame, boxes in zip(vid_proto['frames'], all_boxes):
        print "Detecting in frame {}, {} boxes...".format(
            frame['frame'], len(boxes))
        img = imread(os.path.join(root, frame['path']))
        for bbox in boxes:
            det = {}
            det['frame'] = frame['frame']
            det['bbox'] = bbox.tolist()
            det['hash'] = bbox_hash(det_proto['video'], frame['frame'], bbox)
            scores = det_fun(img, bbox, net)
            det['scores'] = score_proto(class_names, scores)
            detections.append(det)
    det_proto['detections'] = detections
    return det_proto


def det_vid_score(vid_proto, det_fun, net, box_proto=None,
                    class_names=imagenet_vdet_classes):
    if box_proto:
        return det_vid_with_box(vid_proto, box_proto, det_fun, net, class_names)
    else:
        return det_vid_without_box(vid_proto, det_fun, net, class_names)


def apply_vid_nms(det_proto, class_index, thres=0.3):
    print 'Apply NMS on video: {}'.format(det_proto['video'])
    new_det = {}
    new_det['video'] = det_proto['video']
    boxes = np.asarray([[det['frame'],]+det['bbox']+[det_score(det, class_index),]
             for det in det_proto['detections']], dtype='float32')
    keep = vid_nms(boxes, thresh=0.3)
    new_det['detections'] = copy.copy([det_proto['detections'][i] for i in keep])
    print "{} / {} windows kept.".format(len(new_det['detections']),
                                         len(det_proto['detections']))
    return new_det


