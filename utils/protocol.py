#!/usr/bin/env python
#
# Manipulate protocols
# Copyright by Kai KANG (myfavouritekk@gmail.com)
#

"""
Protocols
------------------

- bounding boxes: [x1, y1, x2, y2]
- videos: .vid
    ```json
        {
            "video": "video_name",
            "frames": [
                {
                    "frame": 1,
                    "path": path1
                },
                {
                    "frame": 2,
                    "path": path2
                },
                {
                    // ...
                }
            ],
            "root_path": root_path
        }
    ```
- tracklets: .track
    ```json
        {
            "video": "video_name",
            "method": "tracking_method_name",
            "tracks": [
                [
                    {
                        "frame": 1,
                        "bbox": [x1, y1, x2, y2],
                        "score": score1
                        "hash": md5("video_name_frameid_x1_y1_x2_y2")
                    },
                    {
                        "frame": 2,
                        "bbox": [x1, y1, x2, y2],
                        "score": score2
                        "hash": md5("video_name_frameid_x1_y1_x2_y2")
                    }
                ],  // tracklet 1
                [
                    // tracklet 2
                ]
                // ...
            ]
        }
    ```
- box_files: .box
    ```json
        {
            "video": "video_name",
            "boxes": [
                {
                    "frame": 1,
                    "bbox": [x1, y1, x2, y2]
                    "hash": md5("video_name_frameid_x1_y1_x2_y2")
                },
                {
                    //...
                }
            ]
        }
    ```
- detections: .det
    ```json
        {
            "video": "video_name",
            "detections": [
                {
                    "frame": 1,
                    "bbox": [x1, y1, x2, y2],
                    "hash": md5("video_name_frameid_x1_y1_x2_y2"),
                    "scores": [
                        {
                            "class": "class1",
                            "class_index": idx1,
                            "score": score1
                        },
                        {
                            "class": "class2",
                            "class_index": idx2,
                            "score": score2
                        }
                    ]
                },
                {
                    "frame": 1,
                    "bbox": [x1, y1, x2, y2],
                    "hash": md5("video_name_frameid_x1_y1_x2_y2"),
                    "scores": [
                        // ...
                    ]
                }
                // ...
            ]
        }
    ```
- annotation: .annot
    ```json
        {
            "video": "video_name",
            "annotations": [
                {
                    "id": "track_id"
                    "track":[
                        {
                            "frame": 1,
                            "bbox": [x1, y1, x2, y2],
                            "name": WNID,
                            "class": "class1",
                            "class_index": idx1
                        },
                        {
                            "frame": 2,
                            "bbox": [x1, y1, x2, y2],
                            "name": WNID,
                            "class": "class1",
                            "class_index": idx1
                        }
                    ]
                },  // tracklet 1
                {
                    // tracklet 2
                }
                // ...
            ]
        }
    ```
"""

from common import isimg, sort_nicely
from log import logging
import json
import hashlib
import os
import copy
import numpy as np

##########################################
## General Protocol Manipulation
##########################################

def proto_load(file_path):
    with open(file_path, 'r') as f:
        obj = json.load(f)
    return obj


def proto_dump(obj, file_path):
    with open(file_path, 'w') as f:
        json.dump(obj, f, indent=2)


##########################################
## Video Protocol
##########################################

def vid_proto_from_dir(root_dir, vid_name=None):
    vid = {}
    vid['root_path'] = root_dir
    frames = []
    frame_list = [i for i in os.listdir(root_dir) if isimg(i)]
    sort_nicely(frame_list)
    for index, path in enumerate(frame_list):
        frames.append({'frame': index+1,
                       'path': path})
    vid['frames'] = frames
    if not vid_name:
        # infer video namke from root_dir if not provided
        vid_name = stem(root_dir)
    vid['video'] = vid_name
    return vid


def frame_path_at(vid_proto, frame_id):
    frame = [frame for frame in vid_proto['frames'] if frame['frame'] == frame_id][0]
    return str(os.path.join(vid_proto['root_path'], frame['path']))


def frame_path_before(vid_proto, frame_id):
    frames = [frame for frame in vid_proto['frames'] if frame['frame'] <= frame_id]
    return [str(os.path.join(vid_proto['root_path'], frame['path'])) \
                for frame in frames]


def frame_path_after(vid_proto, frame_id):
    frames = [frame for frame in vid_proto['frames'] if frame['frame'] >= frame_id]
    return [str(os.path.join(vid_proto['root_path'], frame['path'])) \
                for frame in frames]


def sample_vid_proto(vid_proto, stride=10):
    new_vid = {}
    new_vid['video'] = vid_proto['video']
    new_vid['root_path'] = vid_proto['root_path']
    idx = np.arange(0, len(vid_proto['frames']), stride)
    logging.info("Sampling video by 1 / {}.".format(stride))
    new_vid['frames'] = [vid_proto['frames'][i] for i in idx]
    return new_vid


##########################################
## Detection Protocol
##########################################

def empty_det_from_box(box_proto):
    det_proto = {}
    det_proto['video'] = box_proto['video']
    detections = box_proto['boxes']
    for i in detections:
        i['scores'] = []
    det_proto['detections'] = detections
    return det_proto


def score_proto(class_names, scores):
    sc_proto = []
    if type(scores) is not list:
        # numpy array
        scores = scores.tolist()
    for idx, (cls_name, score) in enumerate(zip(class_names, scores)):
        sc_proto.append(
            {
                'class': cls_name,
                'class_index': idx,
                'score': score
            }
        )
    return sc_proto


def det_score(detection, class_index):
    for score in detection['scores']:
        if score['class_index'] == class_index:
            return score['score']
    return float('-inf')


def top_detections(det_proto, top_num, class_index):
    if len(det_proto['detections']) < top_num:
        return copy.copy(det_proto)
    new_det = {}
    new_det['video'] = det_proto['video']
    sorted_det = copy.copy(det_proto['detections'])
    sorted_det = sorted(sorted_det,
            key=lambda x: det_score(x, class_index), reverse=True)
    new_det['detections'] = sorted_det[:top_num]
    return new_det

def frame_top_detections(det_proto, top_num, class_index):
    new_det = {}
    new_det['video'] = det_proto['video']
    new_det['detections'] = []
    frame_idx = list(set([det['frame'] for det in det_proto['detections']]))
    for frame_id in frame_idx:
        cur_dets = copy.copy([det for det in det_proto['detections'] if det['frame'] == frame_id])
        cur_dets = sorted(cur_dets,
            key=lambda x: det_score(x, class_index), reverse=True)
        new_det['detections'].extend(cur_dets[:top_num])
    return new_det


##########################################
## Proposal Protocol
##########################################

def boxes_proto_from_boxes(frame_idx_list, boxes_list, video_name):
    boxes_proto = []
    for frame_idx, boxes in zip(frame_idx_list, boxes_list):
        for bbox in boxes:
            boxes_proto.append(
                    {
                        'frame': int(frame_idx),
                        'bbox': map(int, bbox),
                        'hash': bbox_hash(video_name, frame_idx, bbox)
                    }
                )
    return boxes_proto


def bbox_hash(video_name, frame_id, bbox):
    return hashlib.md5('{}_{}_{}_{}_{}_{}'.format(
            video_name, frame_id,
            bbox[0], bbox[1], bbox[2], bbox[3])).hexdigest()


##########################################
## Tracking Protocol
##########################################

def tracks_proto_from_boxes(boxes, video_name):
    tracks_proto = []
    started = False
    for frame_idx, bbox in enumerate(boxes, start=1):
        if np.any(np.isnan(bbox)): # invalid boxes
            if started: # end old track
                tracks_proto.append(track)
                started = False
            continue
        if not started: # start new track
            started = True
            track = []

        track.append(
                {
                    'frame': frame_idx,
                    'bbox': [int(cor) for cor in bbox[0:4]],
                    'hash': bbox_hash(video_name, frame_idx, bbox),
                    'score': float(bbox[4])
                }
            )
    if started:
        tracks_proto.append(track)
    return tracks_proto


def track_box_at_frame(tracklet, frame_id):
    for box in tracklet:
        if box['frame'] == frame_id:
            return box['bbox']
    return None
