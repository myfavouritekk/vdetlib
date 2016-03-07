#!/usr/bin/env python
import os
import numpy as np
import copy
from dataset import imagenet_vdet_classes
from proposal import vid_proposals
from ..utils.protocol import empty_det_from_box, score_proto, bbox_hash, det_score, boxes_at_frame, frame_path_at
from ..utils.common import imread
from ..utils.log import logging
from ..utils.timer import Timer
from ..utils.cython_nms import vid_nms
import heapq

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


def fast_rcnn_det_vid(net, vid_proto, box_proto, det_fun,
        class_names=imagenet_vdet_classes, max_per_image=100, thresh=0.05):
    """Test a Fast R-CNN network on a video protocol."""
    num_images = len(vid_proto['frames'])

    num_classes = len(class_names)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(num_classes)]

    # timers
    _t = {'im_detect' : Timer(), 'misc' : Timer()}

    for i, frame in enumerate(vid_proto['frames']):
        im = imread(frame_path_at(vid_proto, frame['frame']))
        _t['im_detect'].tic()
        orig_boxes = np.array([box['bbox'] for box in \
            boxes_at_frame(box_proto, frame['frame'])])
        scores, boxes = det_fun(net, im, orig_boxes)
        _t['im_detect'].toc()


        _t['misc'].tic()
        for j in xrange(1, num_classes):
            inds = np.where(scores[:, j] > thresh)[0]
            cls_scores = scores[inds, j]
            cls_boxes = boxes[inds, j*4:(j+1)*4]
            if len(cls_scores) > max_per_image:
                top_inds = np.argsort(-cls_scores)[:max_per_image]
                cls_scores = cls_scores[top_inds]
                cls_boxes = cls_boxes[top_inds, :]
            all_boxes[j][i] = \
                    np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                    .astype(np.float32, copy=False)
        _t['misc'].toc()

        print 'im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
              .format(i + 1, num_images, _t['im_detect'].average_time,
                      _t['misc'].average_time)

    return all_boxes
