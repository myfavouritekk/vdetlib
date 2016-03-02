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
                        "score": score1,
                        "hash": md5("video_name_frameid_x1_y1_x2_y2"),
                        "anchor": int
                    },
                    {
                        "frame": 2,
                        "bbox": [x1, y1, x2, y2],
                        "score": score2,
                        "hash": md5("video_name_frameid_x1_y1_x2_y2"),
                        "anchor": int
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
- score: .score
    ```json
        {
            "video": "video_name",
            "method": "scoring_method_name",
            "tubelets": [
                {
                    "gt": bool,
                    "class": class_name,
                    "class_index": class_index,
                    "boxes": [
                        {
                            "frame": 1,
                            "bbox": [x1, y1, x2, y2],
                            "track_score": track_score1,
                            "det_score": det_score1,
                            "conv_score": conv_score1,
                            "all_score": [cls1_sc, cls2_sc, ...],
                            "feat": [feat1, feat2, ...],
                            "hash": md5("video_name_frameid_x1_y1_x2_y2"),
                            "anchor": int,
                            "gt_overlap": iou_value
                        },
                        {
                            "frame": 2,
                            "bbox": [x1, y1, x2, y2],
                            "track_score": track_score2,
                            "det_score": det_score2,
                            "conv_score": conv_score2,
                            "all_score" : [cls1_sc, cls2_sc, ...],
                            "feat": [feat1, feat2, ...],
                            "hash": md5("video_name_frameid_x1_y1_x2_y2"),
                            "anchor": int,
                            "gt_overlap": iou_value
                        }
                    ]
                },  // tubelet 1
                {
                    // tubelet 2
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
                            "class_index": idx1,
                            "generated": bool,
                            "occluded": bool,
                            "frame_size": [height, width]
                        },
                        {
                            "frame": 2,
                            "bbox": [x1, y1, x2, y2],
                            "name": WNID,
                            "class": "class1",
                            "class_index": idx1,
                            "generated": bool,
                            "occluded": bool
                            "frame_size": [height, width]
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

from common import isimg, sort_nicely, iou
from ..vdet.dataset import imagenet_vdet_classes
from log import logging
import json
import hashlib
import os
import copy
import numpy as np
import scipy.io as sio
import gzip

##########################################
## General Protocol Manipulation
##########################################

def proto_load(file_path):
    # load .gz version if exists
    # AD_HOC implementation
    if os.path.isfile(file_path + '.gz'):
        file_path += '.gz'
    if os.path.splitext(file_path)[1] == '.gz':
        with gzip.GzipFile(file_path) as f:
            obj = json.loads(f.read())
    else:
        with open(file_path, 'r') as f:
            obj = json.load(f)
    return obj


def proto_dump(obj, file_path):
    if os.path.splitext(file_path)[1] == '.gz':
        try:
            with gzip.GzipFile(file_path, 'w', 1) as f:
                f.write(json.dumps(obj, indent=2))
                return
        except OverflowError:
            print "Buffer exceeds 2GB, fallback to regular file."
            if os.path.isfile(file_path):
                os.remove(file_path)
            file_path = os.path.splitext(file_path)[0]

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

def path_to_index(vid_proto, path):
    for frame in vid_proto['frames']:
        if frame['path'].startswith(path):
            return frame['frame']
    return None


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


def boxes_at_frame(box_proto, frame_id):
    boxes = []
    for box in box_proto['boxes']:
        if box['frame'] == frame_id:
            boxes.append(copy.copy(box))
    return boxes

##########################################
## Tracking Protocol
##########################################

def tracks_proto_from_boxes(boxes, video_name, anchor, start_frame=1, step=1):
    tracks_proto = []
    started = False
    for box_idx, bbox in enumerate(boxes):
        frame_idx = start_frame + box_idx * step
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
                    'score': float(bbox[4]),
                    'anchor': int(frame_idx - anchor)
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

def track_proto_from_annot_proto(annot_proto):
    vid_name = annot_proto['video']
    track_proto = {}
    track_proto['video'] = vid_name
    track_proto['method'] = 'gt'
    tracks_proto = []
    for annot_track in annot_proto['annotations']:
        cur_track = []
        for annot_box in annot_track['track']:
            cur_track.append(
                {
                    "frame": annot_box['frame'],
                    "bbox": annot_box['bbox'],
                    "score": 1,
                    "anchor": 0,
                    "hash": bbox_hash(vid_name, annot_box['frame'], annot_box['bbox'])
                }
            )
        tracks_proto.append(cur_track)
    track_proto['tracks'] = tracks_proto
    return track_proto

##########################################
## Scoring Protocol
##########################################
def tubelets_proto_from_tracks_proto(tracks_proto, class_index):
    tubelet_proto = []
    for track in tracks_proto:
        tubelet = {}
        tubelet['gt'] = 0
        tubelet['class_index'] = class_index
        tubelet['class'] = imagenet_vdet_classes[class_index]
        tubelet_boxes = []
        for box in track:
            tubelet_box = copy.copy(box)
            tubelet_box['track_score'] = tubelet_box['score']
            tubelet_box['det_score'] = -1e5
            del tubelet_box['score']
            tubelet_boxes.append(tubelet_box)
        tubelet['boxes'] = tubelet_boxes
        tubelet_proto.append(tubelet)
    return tubelet_proto


def tubelets_overlap(tubelets_proto, annot_proto, class_idx):
    for tubelet in tubelets_proto:
        class_index = tubelet['class_index']
        ious = []
        # for each tubelet_box find the best gt_overlap
        for tubelet_box in tubelet['boxes']:
            tubelet_box['gt_overlap'] = 0
            for annot_track in annot_proto['annotations']:
                for annot_box in annot_track['track']:
                    if annot_box['class_index'] != class_index:
                        # only need to check class of first annot_box, so we break
                        break
                    if tubelet_box['frame'] == annot_box['frame']:
                        cur_iou = iou([annot_box['bbox']],  [tubelet_box['bbox']])
                        # convert ndarray to a scalar
                        cur_iou = float(cur_iou.ravel())
                        if 'gt_overlap' not in tubelet_box or cur_iou > tubelet_box['gt_overlap']:
                            tubelet_box['gt_overlap'] = cur_iou
        ious = [box['gt_overlap'] for box in tubelet['boxes']]
        mean_iou = np.asarray(ious).mean()
        if abs(mean_iou - 1) < np.finfo(float).eps:
            tubelet['gt'] = 1
    return tubelets_proto


def tubelet_box_at_frame(tubelet, frame_id):
    for box in tubelet['boxes']:
        if box['frame'] == frame_id:
            return box['bbox']
    return None

def tubelet_box_proto_at_frame(tubelet, frame_id):
    for box in tubelet['boxes']:
        if box['frame'] == frame_id:
            return box
    return None

def merge_score_protos(proto_1, proto_2, scheme='combine'):
    assert scheme in ['combine', 'max']
    assert proto_1['video'] == proto_2['video']
    new_proto = copy.copy(proto_1)
    if proto_1['method'] != proto_2['method']:
        new_proto['method'] = '_'.join([proto_1['method'], proto_2['method']])
    if scheme == 'combine':
        new_proto['tubelets'].extend(copy.copy(proto_2['tubelets']))
    elif scheme == 'max':
        for tubelet1, tubelet2 in \
            zip(new_proto['tubelets'], proto_2['tubelets']):
            assert tubelet1['gt'] == tubelet2['gt']
            assert tubelet1['class'] == tubelet2['class']
            assert tubelet1['class_index'] == tubelet2['class_index']
            for box1, box2 in zip(tubelet1['boxes'], tubelet2['boxes']):
                assert box1['frame'] == box2['frame']
                assert box1['anchor'] == box2['anchor']
                assert box1['frame'] == box2['frame']
                if box1['det_score'] < box2['det_score']:
                    for key in box1:
                        box1[key] = copy.copy(box2[key])
    return new_proto


def load_frame_to_det(vid_proto, det_dir):
    frame_to_det = {}
    for frame in vid_proto['frames']:
        frame_id = frame['frame']
        basename = os.path.splitext(frame['path'])[0]
        score_file = os.path.join(det_dir, basename + '.mat')
        if not os.path.isfile(score_file):
            score_file = os.path.join(det_dir, frame['path'] + '.mat')
        if os.path.isfile(score_file):
            d = sio.loadmat(score_file)
            frame_to_det[frame_id] = (d['boxes'], d['zs'])
    return frame_to_det

def load_det_info(vid_proto, det_dir):
    # [[frame_id, x1, y1, x2, y2, scores], ...]
    det_info = []
    for frame in vid_proto['frames']:
        frame_id = frame['frame']
        basename = os.path.splitext(frame['path'])[0]
        score_file = os.path.join(det_dir, basename + '.mat')
        if not os.path.isfile(score_file):
            score_file = os.path.join(det_dir, frame['path'] + '.mat')
        if os.path.isfile(score_file):
            d = sio.loadmat(score_file)
            if d['boxes'].size == 0: continue
            for boxes, scores in zip(d['boxes'], d['zs']):
                det_info.append([frame_id,] + boxes.tolist() + scores.tolist())
    return np.asarray(det_info)
