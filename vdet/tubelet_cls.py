#!/usr/bin/env python
from ..utils.protocol import frame_path_at, track_box_at_frame, bbox_hash
from ..utils.common import imread
from ..utils.log import logging
from ..vdet.image_det import googlenet_det
import numpy as np
import sys
import os
sys.path.insert(1, os.path.join(os.path.dirname(__file__),
    '../../External/fast-rcnn/lib/'))
from fast_rcnn.test import im_detect


def fast_rcnn_cls(video_proto, track_proto, net, class_idx):
    new_tracks = [[] for _ in track_proto['tracks']]
    for frame in video_proto['frames']:
        frame_id = frame['frame']
        img = imread(frame_path_at(video_proto, frame_id))
        boxes = [track_box_at_frame(tracklet, frame_id) \
                 for tracklet in track_proto['tracks']]
        valid_boxes = np.asarray([box for box in boxes if box is not None])
        valid_index = [i for i in len(boxes) if boxes[i] is not None]
        scores, pred_boxes = im_detect(net, img, valid_boxes)
        for score, box, track_id in zip(scores, pred_boxes, valid_index):
            new_tracks[track_id].append(
                {
                    "frame": frame_id,
                    "bbox": list(box),
                    "score": score[class_idx],
                    "hash": bbox_hash(video_proto['video'], frame_id, box)
                })
    return new_tracks


def googlenet_cls(video_proto, track_proto, net, class_idx):
    new_tracks = [[] for _ in track_proto['tracks']]
    logging.info("Classifying {}...".format(video_proto['video']))
    for frame in video_proto['frames']:
        frame_id = frame['frame']
        img = imread(frame_path_at(video_proto, frame_id))
        boxes = [track_box_at_frame(tracklet, frame_id) \
                 for tracklet in track_proto['tracks']]
        valid_boxes = np.asarray([box for box in boxes if box is not None])
        valid_index = [i for i, box in enumerate(boxes) if box is not None]
        logging.info("frame {}: {} boxes".format(frame_id, len(valid_index)))
        for box, track_id in zip(valid_boxes, valid_index):
            scores = googlenet_det(img, box, net)
            new_tracks[track_id].append(
                {
                    "frame": frame_id,
                    "bbox": list(box),
                    "score": scores[class_idx],
                    "hash": bbox_hash(video_proto['video'], frame_id, box)
                })
    return new_tracks


def classify_tracks(video_proto, track_proto, cls_method, net, class_idx):
    assert video_proto['video'] == track_proto['video']
    cls_track = {}
    cls_track['video'] = video_proto['video']
    cls_track['method'] = cls_method.__name__
    cls_track['tracks'] = cls_method(video_proto, track_proto, net, class_idx)
    return cls_track

