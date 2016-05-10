#!/usr/bin/env python
from ..utils.protocol import frame_path_at, track_box_at_frame, bbox_hash, \
    tubelets_overlap, tubelets_proto_from_tracks_proto, tubelet_box_at_frame, det_score, tubelet_box_proto_at_frame
from ..utils.common import imread, svm_from_rcnn_model, iou
from ..utils.log import logging
from ..vdet.image_det import googlenet_det, googlenet_features, svm_scores
from ..vdet.dataset import imagenet_vdet_classes, index_vdet_to_det
import numpy as np
import sys
import os
import copy
from scipy.interpolate import interp1d
from collections import defaultdict

def score_conv_cls(score_proto, net):
    new_score_proto = copy.copy(score_proto)
    print "{}: {} tubelet(s).".format(score_proto['video'], len(new_score_proto['tubelets']))
    for tubelet in new_score_proto['tubelets']:
        track = {}
        track['length'] = len(tubelet['boxes'])
        track['gt'] = tubelet['gt']
        track['mean_iou'] = np.mean([map(lambda x:x['gt_overlap'],
                                  tubelet['boxes'])])
        track['det_scores'] = map(lambda x:x['det_score'],
                                  tubelet['boxes'])
        track['track_scores'] = map(lambda x:x['track_score'],
                                  tubelet['boxes'])
        track['anchors'] = map(lambda x:x['anchor'] * 1. / track['length'],
                                  tubelet['boxes'])
        track['abs_anchors'] = map(abs, track['anchors'])
        track['gt_overlaps'] = map(lambda x:x['gt_overlap'],
                                  tubelet['boxes'])
        track['labels'] = [1 if iou >= 0.5 else 0 for iou in track['gt_overlaps']]

        # skip memory heavy features if possible
        if 'all_scores' in net.blobs.keys():
            track['all_scores'] = map(lambda x:x['all_score'],
                                  tubelet['boxes'])
        if 'feats' in net.blobs.keys():
            track['feats'] = map(lambda x:x['feat'],
                                  tubelet['boxes'])

        for blob_name in set(net.blobs.keys()).intersection(set(track.keys())):
            num_channels = net.blobs[blob_name].shape[1]
            net.blobs[blob_name].reshape(1, num_channels, 1, track['length'])
            net.blobs[blob_name].data[...] = np.asarray(track[blob_name], dtype='float32')
        blobs_out = net.forward()
        probs = blobs_out['probs'][:, 1,:]
        for box, prob in zip(tubelet['boxes'], probs.ravel()):
            box['conv_score'] = float(prob)
    return new_score_proto

def fast_rcnn_cls(video_proto, track_proto, net, class_idx):
    sys.path.insert(1, os.path.join(os.path.dirname(__file__),
        '../../External/fast-rcnn/lib/'))
    sys.path.insert(1, os.path.join(os.path.dirname(__file__),
        '../../External/fast-rcnn/caffe-fast-rcnn/python'))
    from fast_rcnn.test import im_detect
    new_tracks = [[] for _ in track_proto['tracks']]
    for frame in video_proto['frames']:
        frame_id = frame['frame']
        img = imread(frame_path_at(video_proto, frame_id))
        boxes = [track_box_at_frame(tracklet, frame_id) \
                 for tracklet in track_proto['tracks']]
        valid_boxes = np.asarray([box for box in boxes if box is not None])
        valid_index = [i for i in xrange(len(boxes)) if boxes[i] is not None]
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


def rcnn_scoring(vid_proto, track_proto, net, class_idx, rcnn_model,
        save_feat=False, save_all_sc=False):
    svm_model = svm_from_rcnn_model(rcnn_model)
    tubelets_proto = tubelets_proto_from_tracks_proto(track_proto['tracks'], class_idx)
    logging.info("Scoring {} for {}...".format(vid_proto['video'],
                 imagenet_vdet_classes[class_idx]))
    for frame in vid_proto['frames']:
        frame_id = frame['frame']
        img = imread(frame_path_at(vid_proto, frame_id))
        boxes = [tubelet_box_at_frame(tubelet, frame_id) \
                 for tubelet in tubelets_proto]
        valid_boxes = np.asarray([box for box in boxes if box is not None])
        valid_index = [i for i, box in enumerate(boxes) if box is not None]
        logging.info("frame {}: {} boxes".format(frame_id, len(valid_index)))
        if len(valid_index) == 0:
            continue
        features = googlenet_features(img, valid_boxes, net, 'pool5')
        scores = svm_scores(features, svm_model)
        if scores.shape[1] == 200:
            cls_scores = scores[:, index_vdet_to_det[class_idx] - 1]
        else:
            raise
        for score, tubelet_id, feat, all_score in \
                zip(cls_scores, valid_index, features, scores):
            cur_box = [box for box in tubelets_proto[tubelet_id]['boxes'] \
                if box['frame'] == frame_id]
            assert len(cur_box) == 1
            cur_box[0]['det_score'] = score
            if save_feat:
                cur_box[0]['feat'] = feat.ravel().tolist()
            if save_all_sc:
                cur_box[0]['all_score'] = all_score.ravel().tolist()
    return tubelets_proto

def sampling_boxes(orig_box, num, ratio = 0.05, return_orig=True):
    h, w = orig_box[3] - orig_box[1], orig_box[2] - orig_box[0]
    offsets = np.random.uniform(-ratio, ratio, [num, 4]) * [w, h, w, h]
    if not return_orig:
        return orig_box + offsets
    else:
        return np.vstack((orig_box, orig_box+offsets))

def rcnn_sampling_scoring(vid_proto, track_proto, net, class_idx, rcnn_model,
        samples_per_box = 32, ratio = 0.05,
        save_feat=False, save_all_sc=False):
    svm_model = svm_from_rcnn_model(rcnn_model)
    tubelets_proto = tubelets_proto_from_tracks_proto(track_proto['tracks'], class_idx)
    logging.info("Scoring {} for {}...".format(vid_proto['video'],
                 imagenet_vdet_classes[class_idx]))
    for frame in vid_proto['frames']:
        frame_id = frame['frame']
        img = imread(frame_path_at(vid_proto, frame_id))
        boxes = [tubelet_box_at_frame(tubelet, frame_id) \
                 for tubelet in tubelets_proto]
        valid_boxes = np.asarray([box for box in boxes if box is not None])
        valid_index = [i for i, box in enumerate(boxes) if box is not None]
        logging.info("frame {}: {} boxes".format(frame_id, len(valid_index)))
        if len(valid_index) == 0:
            continue

        # sample nearby boxes to increase spatial robustness
        sampled_boxes = np.vstack([sampling_boxes(box, samples_per_box, ratio) \
                for box in valid_boxes])
        features = googlenet_features(img, sampled_boxes, net, 'pool5')
        scores = svm_scores(features, svm_model)
        if scores.shape[1] == 200:
            cls_scores = scores[:, index_vdet_to_det[class_idx] - 1]
            cls_scores = cls_scores.reshape((len(valid_index), -1))
            max_scores = cls_scores.max(axis=1)
            # extract cooresponding features and all class scores
            max_idx = np.argmax(cls_scores, axis=1)
        else:
            raise

        # extract features and scores of box with maximum score
        features = features.reshape((len(valid_index), samples_per_box+1, -1))
        features = features[xrange(len(valid_index)), max_idx,:]
        scores = scores.reshape((len(valid_index), samples_per_box+1, -1))
        scores = scores[xrange(len(valid_index)), max_idx,:]
        sampled_boxes = sampled_boxes.reshape((len(valid_index), samples_per_box+1, -1))
        boxes = sampled_boxes[xrange(len(valid_index)), max_idx,:]
        for score, tubelet_id, feat, all_score, max_box in \
                zip(max_scores, valid_index, features, scores, boxes):
            cur_box = [box for box in tubelets_proto[tubelet_id]['boxes'] \
                if box['frame'] == frame_id]
            assert len(cur_box) == 1
            cur_box[0]['det_score'] = score
            cur_box[0]['bbox'] = max_box.tolist()
            if save_feat:
                cur_box[0]['feat'] = feat.ravel().tolist()
            if save_all_sc:
                cur_box[0]['all_score'] = all_score.ravel().tolist()
    return tubelets_proto

def rcnn_sampling_dets_scoring(vid_proto, track_proto, det_proto,
        net, class_idx, rcnn_model, overlap_thres=0.7,
        save_feat=False, save_all_sc=False):
    svm_model = svm_from_rcnn_model(rcnn_model)
    tubelets_proto = tubelets_proto_from_tracks_proto(track_proto['tracks'], class_idx)
    logging.info("Scoring {} for {}...".format(vid_proto['video'],
                 imagenet_vdet_classes[class_idx]))
    for frame in vid_proto['frames']:
        frame_id = frame['frame']
        img = imread(frame_path_at(vid_proto, frame_id))
        boxes = [tubelet_box_at_frame(tubelet, frame_id) \
                 for tubelet in tubelets_proto]
        valid_boxes = np.asarray([box for box in boxes if box is not None])
        valid_index = [i for i, box in enumerate(boxes) if box is not None]
        logging.info("frame {}: {} boxes".format(frame_id, len(valid_index)))
        if len(valid_index) == 0:
            continue
        # compute rcnn scores
        features = googlenet_features(img, valid_boxes, net, 'pool5')
        scores = svm_scores(features, svm_model)
        if scores.shape[1] == 200:
            cls_scores = scores[:, index_vdet_to_det[class_idx] - 1]
        else:
            raise

        # find all detection proposal boxes in current frame
        dets = [det for det in det_proto['detections'] if det['frame']==frame_id]
        det_boxes = np.asarray(map(lambda x:x['bbox'], dets))
        det_scores = np.asarray(map(lambda x:det_score(x, class_idx), dets))

        for score, tubelet_id, feat, all_score in \
                zip(cls_scores, valid_index, features, scores):
            cur_box = [box for box in tubelets_proto[tubelet_id]['boxes'] \
                if box['frame'] == frame_id]
            assert len(cur_box) == 1
            # calculate overlaps with all det_boxes in current frame
            if len(det_boxes) > 0:
                overlaps = iou([cur_box[0]['bbox']], det_boxes)
                conf_idx = (overlaps > overlap_thres).ravel()
            else:
                conf_idx = [False]
            if np.any(conf_idx):
                conf_boxes = det_boxes[conf_idx]
                conf_scores = det_scores[conf_idx]
                max_idx = np.argmax(conf_scores)
                max_score = conf_scores[max_idx]
                max_box = conf_boxes[max_idx].tolist()
            else:
                max_score = -np.inf
            if max_score > score:
                cur_box[0]['det_score'] = max_score
                cur_box[0]['bbox'] = max_box
                max_feat = googlenet_features(img, [max_box], net, 'pool5')
                max_all_scores = svm_scores(max_feat, svm_model)
                if save_feat:
                    cur_box[0]['feat'] = max_feat.ravel().tolist()
                if save_all_sc:
                    cur_box[0]['all_score'] = max_all_scores.ravel().tolist()
            else:
                cur_box[0]['det_score'] = score
                if save_feat:
                    cur_box[0]['feat'] = feat.ravel().tolist()
                if save_all_sc:
                    cur_box[0]['all_score'] = all_score.ravel().tolist()
    return tubelets_proto


def scoring_tracks(vid_proto, track_proto, annot_proto,
        sc_method, net, class_idx):
    assert vid_proto['video'] == track_proto['video']
    score_proto = {}
    score_proto['video'] = vid_proto['video']
    score_proto['method'] = sc_method.__name__
    tubelets_proto = sc_method(vid_proto, track_proto, net, class_idx)
    if annot_proto is not None:
        tubelets_proto = tubelets_overlap(tubelets_proto, annot_proto, class_idx)
    score_proto['tubelets'] = tubelets_proto
    return score_proto


def classify_tracks(video_proto, track_proto, cls_method, net, class_idx):
    assert video_proto['video'] == track_proto['video']
    cls_track = {}
    cls_track['video'] = video_proto['video']
    cls_track['method'] = cls_method.__name__
    cls_track['tracks'] = cls_method(video_proto, track_proto, net, class_idx)
    return cls_track

def do_score_completion(score_proto):
    for tubelet in score_proto['tubelets']:
        # deal with -100000
        boxes = tubelet['boxes']
        for i, box in enumerate(boxes):
            if box['det_score'] > -10: continue
            j = i
            while j < len(boxes) and boxes[j]['det_score'] <= -10:
                j += 1
            if i == 0:
                for k in xrange(i, j):
                    boxes[k]['det_score'] = boxes[j]['det_score']
            elif j == len(boxes):
                for k in xrange(i, j):
                    boxes[k]['det_score'] = boxes[i-1]['det_score']
            else:
                l = boxes[i-1]['det_score']
                r = boxes[j]['det_score']
                for k in xrange(i, j):
                    boxes[k]['det_score'] = l + (r - l) * (k - i + 1) / (j - i + 1)

def dets_spatial_max_pooling(vid_proto, track_proto, det_proto, class_idx, overlap_thres=0.7):
    assert vid_proto['video'] == track_proto['video']
    score_proto = {}
    score_proto['video'] = vid_proto['video']
    score_proto['method'] = "spatial_max_pooling_IOU_{}".format(overlap_thres)

    tubelets_proto = tubelets_proto_from_tracks_proto(track_proto['tracks'], class_idx)
    logging.info("Sampling dets in {} for {}...".format(vid_proto['video'],
                 imagenet_vdet_classes[class_idx]))

    # build dict to fast indexing
    frame_to_tubelets_idx = defaultdict(list)
    for i, tubelet in enumerate(tubelets_proto):
        for j, box in enumerate(tubelet['boxes']):
            frame_to_tubelets_idx[box['frame']].append((i, j))
    frame_to_det_idx = defaultdict(list)
    for i, det in enumerate(det_proto['detections']):
        frame_to_det_idx[det['frame']].append(i)

    for frame in vid_proto['frames']:
        frame_id = frame['frame']
        det_idx = frame_to_det_idx[frame_id]
        if len(det_idx) == 0: continue
        det_boxes = np.asarray([det_proto['detections'][i]['bbox'] for i in det_idx])
        det_scores = np.asarray([det_proto['detections'][i]['scores'][class_idx - 1]['score'] for i in det_idx])
        for i, j in frame_to_tubelets_idx[frame_id]:
            cur_box = tubelets_proto[i]['boxes'][j]
            if cur_box is None: continue
            overlaps = iou([cur_box['bbox']], det_boxes)
            overlap_idx = (overlaps > overlap_thres).ravel()
            if np.any(overlap_idx):
                conf_boxes = det_boxes[overlap_idx]
                conf_scores = det_scores[overlap_idx]
                max_idx = np.argmax(conf_scores)
                max_score = conf_scores[max_idx]
                max_box = conf_boxes[max_idx].tolist()
            else:
                print "Warning: Tubelet {} has no overlapping dets (IOU > {}).".format(
                    i, overlap_thres)
                max_score = -1e5
                max_box = cur_box['bbox']
            cur_box['det_score'] = float(max_score)
            cur_box['bbox'] = max_box
    score_proto['tubelets'] = tubelets_proto
    do_score_completion(score_proto)
    return score_proto


def anchor_propagate(vid_proto, track_proto, det_proto, class_idx):
    assert vid_proto['video'] == track_proto['video']
    score_proto = {}
    score_proto['video'] = vid_proto['video']
    score_proto['method'] = "anchor_propagate"
    tubelets_proto = tubelets_proto_from_tracks_proto(track_proto['tracks'], class_idx)
    logging.info("Propagating anchor scores in {} for {}...".format(vid_proto['video'],
                 imagenet_vdet_classes[class_idx]))
    frame_to_det_idx = defaultdict(list)
    for i, det in enumerate(det_proto['detections']):
        frame_to_det_idx[det['frame']].append(i)
    for tubelet in tubelets_proto:
        # find anchor box
        anchor_box = [box for box in tubelet['boxes'] if box['anchor'] == 0]
        assert len(anchor_box) == 1
        anchor_box = anchor_box[0]

        # get anchor detection score
        anchor_frame = anchor_box['frame']
        det_idx = frame_to_det_idx[anchor_frame]
        det_boxes = np.asarray([det_proto['detections'][i]['bbox'] for i in det_idx])
        det_scores = np.asarray([det_proto['detections'][i]['scores'][class_idx - 1]['score'] for i in det_idx])
        overlaps = iou([anchor_box['bbox']], det_boxes)[0]
        anchor_index = np.argmax(overlaps)
        anchor_score = det_scores[anchor_index]

        # propagate anchor scores
        for box in tubelet['boxes']:
            box['det_score'] = anchor_score
    score_proto['tubelets'] = tubelets_proto
    return score_proto


def score_proto_temporal_maxpool(score_proto, window_size):
    if window_size == 1:
        return score_proto
    if window_size % 2 != 1:
        raise ValueError('Window size must be odd!')
    half_window_size = window_size / 2

    new_score_proto = copy.copy(score_proto)
    new_score_proto['method'] += '_temporal_maxpool_{}'.format(window_size)
    for tubelet in new_score_proto['tubelets']:
        if tubelet['gt'] == 1:
            raise ValueError('Dangerous: Score file contains gt tracks!')

        boxes = tubelet['boxes']
        scores = [boxes[i]['det_score'] for i in range(len(boxes))]
        scores = np.array(scores)
        scores = np.pad(scores, (0, window_size-1), 'constant', constant_values=(-1e+5,-1e+5))
        scores = np.tile(scores.T, (window_size, 1))

        for roll in range(window_size):
            scores[roll] = np.roll(scores[roll], roll)

        max_score = scores.max(0)
        max_score = max_score[half_window_size:-half_window_size]

        for i in range(len(boxes)):
            tubelet['boxes'][i]['det_score'] = float(max_score[i])

    return new_score_proto

def extrap1d(interpolator):
    xs = interpolator.x
    ys = interpolator.y

    def pointwise(x):
        if x < xs[0]:
            return ys[0]+(x-xs[0])*(ys[1]-ys[0])/(xs[1]-xs[0])
        elif x > xs[-1]:
            return ys[-1]+(x-xs[-1])*(ys[-1]-ys[-2])/(xs[-1]-xs[-2])
        else:
            return interpolator(x)

    return pointwise

def score_proto_interpolation(score_proto, vid_proto):
    '''Perform interpolation on score protocols is only part of
       the tracks are available'''
    new_score_proto = {}
    new_score_proto['video'] = score_proto['video']
    new_score_proto['method'] = score_proto['method'] + '_interpolation'
    tubelets_proto = []
    idx_fun = lambda x:x['frame']
    funcs = {
        'x1': lambda x:x['bbox'][0],
        'y1': lambda x:x['bbox'][1],
        'x2': lambda x:x['bbox'][2],
        'y2': lambda x:x['bbox'][3],
        'det_score': lambda x:x['det_score'],
        'anchor': lambda x:x['anchor']
    }

    max_frames = len(vid_proto['frames'])

    for tubelet in score_proto['tubelets']:
        if tubelet['gt'] == 1:
            raise ValueError('Dangerous: Score file contains gt tracks!')
        if len(tubelet['boxes']) < 2:
            tubelets_proto.append(copy.copy(tubelet))
            continue

        # Generate interpolation function for each field
        truth_idx = map(idx_fun, tubelet['boxes'])
        interp_funcs = {}
        for field in funcs.keys():
            field_fun = funcs[field]
            truth_values = map(field_fun, tubelet['boxes'])
            interpolator = interp1d(truth_idx, truth_values)
            extrapolator = extrap1d(interpolator)
            interp_funcs[field] = extrapolator
        new_tubelet = {}
        for key in ['gt', 'class', 'class_index']:
            new_tubelet[key] = tubelet[key]
        new_tubelet['boxes'] = []
        min_idx = min(truth_idx)
        max_idx = max(truth_idx)
        # extrapolate first and last frames
        if min_idx == 2:
            min_idx = 1
        if max_idx == max_frames - 1:
            max_idx = max_frames
        for dense_idx in xrange(min_idx, max_idx+1):
            box = {}
            box['frame'] = dense_idx
            box['det_score'] = float(interp_funcs['det_score'](dense_idx))
            box['anchor'] = float(interp_funcs['anchor'](dense_idx))
            x1 = interp_funcs['x1'](dense_idx)
            y1 = interp_funcs['y1'](dense_idx)
            x2 = interp_funcs['x2'](dense_idx)
            y2 = interp_funcs['y2'](dense_idx)
            box['bbox'] = map(float, [x1, y1, x2, y2])
            new_tubelet['boxes'].append(box)
        tubelets_proto.append(new_tubelet)

    new_score_proto['tubelets'] = tubelets_proto
    return new_score_proto


def raw_dets_spatial_max_pooling(vid_proto, track_proto, frame_to_det, class_idx, overlap_thres=0.7):
    assert vid_proto['video'] == track_proto['video']
    score_proto = {}
    score_proto['video'] = vid_proto['video']
    score_proto['method'] = "spatial_max_pooling_IOU_{}".format(overlap_thres)

    tubelets_proto = tubelets_proto_from_tracks_proto(track_proto['tracks'], class_idx)
    logging.info("Sampling dets in {} for {}...".format(vid_proto['video'],
                 imagenet_vdet_classes[class_idx]))

    # build dict to fast indexing
    frame_to_tubelets_idx = defaultdict(list)
    for i, tubelet in enumerate(tubelets_proto):
        for j, box in enumerate(tubelet['boxes']):
            frame_to_tubelets_idx[box['frame']].append((i, j))

    for frame in vid_proto['frames']:
        frame_id = frame['frame']
        if frame_id not in frame_to_det: continue
        det_boxes, det_scores = frame_to_det[frame_id]
        if det_boxes.size == 0: continue
        det_scores = det_scores[:, class_idx - 1].ravel()
        for i, j in frame_to_tubelets_idx[frame_id]:
            cur_box = tubelets_proto[i]['boxes'][j]
            if cur_box is None: continue
            overlaps = iou([cur_box['bbox']], det_boxes)
            overlap_idx = (overlaps > overlap_thres).ravel()
            if np.any(overlap_idx):
                conf_boxes = det_boxes[overlap_idx]
                conf_scores = det_scores[overlap_idx]
                max_idx = np.argmax(conf_scores)
                max_score = conf_scores[max_idx]
                max_box = conf_boxes[max_idx].tolist()
            else:
                print "Warning: Tubelet {} has no overlapping dets (IOU > {}).".format(
                    i, overlap_thres)
                max_score = -1e5
                max_box = cur_box['bbox']
            cur_box['det_score'] = float(max_score)
            cur_box['bbox'] = max_box
    score_proto['tubelets'] = tubelets_proto
    do_score_completion(score_proto)
    return score_proto

