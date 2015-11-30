#!/usr/bin/env python
import os
import sys
from scipy.io import loadmat
import numpy as np
import matlab
import time
import copy
from collections import defaultdict
from operator import itemgetter
from ..utils.protocol import frame_path_after, frame_path_before, tracks_proto_from_boxes
from ..utils.common import matlab_command, matlab_engine, temp_file
from ..utils.cython_nms import track_det_nms
from ..utils.log import logging
import math


def tld_tracker(vid_proto, det):
    script = os.path.join(os.path.dirname(__file__),
        '../../External/tld_matlab/tld_track.m')
    bbox = det['bbox']
    frame_id = det['frame']
    fw_frames = frame_path_after(vid_proto, frame_id)
    bw_frames = frame_path_before(vid_proto, frame_id)[::-1]
    fw_out = temp_file(suffix='.mat')
    bw_out = temp_file(suffix='.mat')
    matlab_command(script, [bbox,] + fw_frames, fw_out)
    matlab_command(script, [bbox,] + bw_frames, bw_out)
    try:
        fw_trk = loadmat(fw_out)['bbox']
    except:
        logging.error("Forward tracking failed.")
        fw_trk = [bbox+[1.]]+[[float('nan')]*5]*(len(fw_frames)-1)

    try:
        bw_trk = loadmat(bw_out)['bbox']
    except:
        logging.error("Backward tracking failed.")
        bw_trk = [[float('nan')]*5]*(len(bw_frames)-1) + [bbox+[1.]]

    os.remove(fw_out)
    os.remove(bw_out)
    bw_trk = bw_trk[::-1]
    if len(fw_trk) > 1:
        trk = np.concatenate((bw_trk, fw_trk[1:]))
    else:
        trk = bw_trk
    tracks_proto = tracks_proto_from_boxes(trk, vid_proto['video'])
    return tracks_proto


def fcn_tracker(vid_proto, anchor_frame_id, bbox, opts):
    # suppress caffe logs
    try:
        orig_loglevel = os.environ['GLOG_minloglevel']
    except KeyError:
        orig_loglevel = '0'
    os.environ['GLOG_minloglevel'] = '2'

    script = os.path.join(os.path.dirname(__file__),
        '../../External/fcn_tracker_matlab/fcn_tracker.m')
    fw_frames = frame_path_after(vid_proto, anchor_frame_id)
    bw_frames = frame_path_before(vid_proto, anchor_frame_id)[::-1]
    if hasattr(opts, 'max_frames') and opts.max_frames is not None:
        num_frames = int(math.ceil((opts.max_frames+1)/2.))
    else:
        num_frames = np.inf
    if hasattr(opts, 'step'):
        step = opts.step
    else:
        step = 1

    # down sample frame rates
    fw_frames = fw_frames[::step]
    bw_frames = bw_frames[::step]
    # track upto maximum frames
    fw_frames = fw_frames[:min(num_frames, len(fw_frames))]
    bw_frames = bw_frames[:min(num_frames, len(bw_frames))]

    tic = time.time()
    fw_trk = matlab_engine(script,
                [matlab.double(bbox),] + fw_frames + [opts.gpu,], opts.engine)
    if fw_trk is None:
        logging.error("Forward tracking failed: {}".format(sys.exc_info()[0]))
        fw_trk = [bbox+[1.]]

    bw_trk = matlab_engine(script,
                [matlab.double(bbox),] + bw_frames + [opts.gpu,], opts.engine)
    if bw_trk is None:
        logging.error("Backward tracking failed: {}".format(sys.exc_info()[0]))
        bw_trk = [bbox+[1.]]

    bw_trk = bw_trk[::-1]
    if len(fw_trk) > 1:
        trk = np.concatenate((bw_trk, fw_trk[1:]))
    else:
        trk = bw_trk
    toc = time.time()
    logging.info("Speed: {:02f} fps".format(len(trk) / (toc-tic)))
    start_frame = anchor_frame_id - step * (len(bw_trk) - 1);
    tracks_proto = tracks_proto_from_boxes(trk, vid_proto['video'],
            anchor_frame_id, start_frame, step)

    # reset log level
    os.environ['GLOG_minloglevel'] = orig_loglevel
    return tracks_proto


def track_from_det(vid_proto, det_proto, track_method):
    assert vid_proto['video'] == det_proto['video']
    track_proto = {}
    track_proto['video'] = vid_proto['video']
    track_proto['method'] = track_method.__name__
    tracks = []
    for idx, det in enumerate(det_proto['detections'], start=1):
        logging.info("tracking top No.{} in {}".format(idx, vid_proto['video']))
        tracks.extend(track_method(vid_proto, det))
    track_proto['tracks'] = tracks
    return track_proto


def greedily_track_from_det(vid_proto, det_proto, track_method,
                            score_fun, opts):
    '''greedily track top detections and supress detections
       that have large overlaps with tracked boxes'''
    if hasattr(opts, 'nms_thres') and opts.nms_thres is not None:
        nms_thres = opts.nms_thres
    else:
        nms_thres = 0.3
    assert vid_proto['video'] == det_proto['video']
    track_proto = {}
    track_proto['video'] = vid_proto['video']
    track_proto['method'] = track_method.__name__

    dets = sorted(det_proto['detections'], key=lambda x:score_fun(x), reverse=True)
    det_info = np.asarray([[det['frame'],] + det['bbox'] + [score_fun(det),]
                           for det in dets], dtype=np.float32)
    frame_to_det_ids = defaultdict(list)
    for i, det in enumerate(dets):
        frame_to_det_ids[det['frame']].append(i)
    keep = [True] * len(dets)
    cur_top_det_id = 0
    tracks = []
    while np.any(keep) and len(tracks) < opts.max_tracks:
        # tracking top detection
        while cur_top_det_id < len(keep) and not keep[cur_top_det_id]:
            cur_top_det_id += 1
        if cur_top_det_id == len(keep): break
        top_det = dets[cur_top_det_id]
        cur_top_det_id += 1
        # stop tracking if confidence too low
        if score_fun(top_det) < opts.thres:
            logging.info("Upon low confidence: total {} tracks".format(len(tracks)))
            break
        # start new track
        logging.info("tracking top No.{} in {}".format(len(tracks), vid_proto['video']))
        anchor_frame_id = top_det['frame']
        anchor_bbox = map(int, top_det['bbox'])
        try:
            new_tracks = track_method(vid_proto, anchor_frame_id, anchor_bbox, opts)
        except:
            import matlab.engine
            try:
                opts.engine.quit()
            except:
                pass
            opts.engine = matlab.engine.start_matlab('-nodisplay -nojvm -nosplash -nodesktop')
            new_tracks = track_method(vid_proto, anchor_frame_id, anchor_bbox, opts)
        tracks.extend(new_tracks)
        # NMS
        logging.info("Applying nms between new tracks ({}) and detections.".format(len(new_tracks)))
        for tracklet in new_tracks:
            for box in tracklet:
                frame_id = box['frame']
                det_ids = frame_to_det_ids[frame_id]
                det_ids = [i for i in det_ids if keep[i]]
                if len(det_ids) == 0: continue
                t = np.asarray([[frame_id,] + box['bbox']], dtype=np.float32)
                d = det_info[det_ids]
                kp = set(track_det_nms(t, d, nms_thres))
                for i, det_id in enumerate(det_ids):
                    if i not in kp:
                        keep[det_id] = False
        logging.info("{} / {} boxes kept.".format(np.sum(keep), len(keep)))
    track_proto['tracks'] = tracks
    return track_proto


def greedily_track_from_raw_dets(vid_proto, det_info, track_method, class_idx, opts):
    '''greedily track top detections and supress detections
       that have large overlaps with tracked boxes'''
    if hasattr(opts, 'nms_thres') and opts.nms_thres is not None:
        nms_thres = opts.nms_thres
    else:
        nms_thres = 0.3
    track_proto = {}
    track_proto['video'] = vid_proto['video']
    track_proto['method'] = track_method.__name__

    det_info = np.asarray(sorted(det_info[:, [0,1,2,3,4,4+class_idx]],
                                 key=itemgetter(5), reverse=True), dtype=np.float32)

    frame_to_det_ids = defaultdict(list)
    for i, det in enumerate(det_info):
        frame_to_det_ids[det[0]].append(i)

    keep = [True] * len(det_info)
    cur_top_det_id = 0
    tracks = []
    while np.any(keep) and len(tracks) < opts.max_tracks:
        # tracking top detection
        while cur_top_det_id < len(keep) and not keep[cur_top_det_id]:
            cur_top_det_id += 1
        if cur_top_det_id == len(keep): break
        top_det = det_info[cur_top_det_id]
        cur_top_det_id += 1
        # stop tracking if confidence too low
        if top_det[-1] < opts.thres:
            logging.info("Upon low confidence: total {} tracks".format(len(tracks)))
            break
        # start new track
        logging.info("tracking top No.{} in {}".format(len(tracks), vid_proto['video']))
        anchor_frame_id = int(top_det[0])
        anchor_bbox = map(int, top_det[1:5])
        try:
            new_tracks = track_method(vid_proto, anchor_frame_id, anchor_bbox, opts)
        except:
            import matlab.engine
            try:
                opts.engine.quit()
            except:
                pass
            opts.engine = matlab.engine.start_matlab('-nodisplay -nojvm -nosplash -nodesktop')
            new_tracks = track_method(vid_proto, anchor_frame_id, anchor_bbox, opts)
        tracks.extend(new_tracks)
        # NMS
        logging.info("Applying nms between new tracks ({}) and detections.".format(len(new_tracks)))
        for tracklet in new_tracks:
            for box in tracklet:
                frame_id = box['frame']
                det_ids = frame_to_det_ids[frame_id]
                det_ids = [i for i in det_ids if keep[i]]
                if len(det_ids) == 0: continue
                t = np.asarray([[frame_id,] + box['bbox']], dtype=np.float32)
                d = det_info[det_ids]
                kp = set(track_det_nms(t, d, nms_thres))
                for i, det_id in enumerate(det_ids):
                    if i not in kp:
                        keep[det_id] = False
        logging.info("{} / {} boxes kept.".format(np.sum(keep), len(keep)))
    track_proto['tracks'] = tracks
    return track_proto
