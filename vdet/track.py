#!/usr/bin/env python
import os
import sys
from scipy.io import loadmat
import numpy as np
import matlab
import time
import copy
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


def fcn_tracker(vid_proto, det, opts):
    # suppress caffe logs
    try:
        orig_loglevel = os.environ['GLOG_minloglevel']
    except KeyError:
        orig_loglevel = '0'
    os.environ['GLOG_minloglevel'] = '2'

    script = os.path.join(os.path.dirname(__file__),
        '../../External/fcn_tracker_matlab/fcn_tracker.m')
    bbox = map(int, det['bbox'])
    frame_id = det['frame']
    fw_frames = frame_path_after(vid_proto, frame_id)
    bw_frames = frame_path_before(vid_proto, frame_id)[::-1]
    if opts.max_frames is not None:
        num_frames = int(math.ceil((opts.max_frames+1)/2.))
    else:
        num_frames = np.inf
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
    start_frame = frame_id - len(bw_trk) + 1;
    tracks_proto = tracks_proto_from_boxes(trk, vid_proto['video'],
            frame_id, start_frame)

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
    assert vid_proto['video'] == det_proto['video']
    track_proto = {}
    track_proto['video'] = vid_proto['video']
    track_proto['method'] = track_method.__name__
    tracks = []
    dets = copy.copy(det_proto['detections'])
    keep = range(len(dets))
    num_tracks = 0
    while len(dets) > 0 and num_tracks < opts.max_tracks:
        # Tracking top detection
        dets = sorted(dets, key=lambda x:score_fun(x), reverse=True)
        topDet = dets[0]
        # stop tracking if confidence too low
        if score_fun(topDet) < opts.thres:
            print "Upon low confidence: total {} tracks".format(num_tracks)
            break
        try:
            track = track_method(vid_proto, dets[0], opts)
        except:
            import matlab.engine
            try:
                opts.engine.quit()
            except:
                pass
            opts.engine = matlab.engine.start_matlab('-nodisplay -nojvm -nosplash -nodesktop')
            track = track_method(vid_proto, dets[0], opts)
        tracks.extend(track)
        num_tracks += 1

        # NMS
        boxes = [[x['frame'],]+x['bbox']+[score_fun(x),] \
                 for x in dets]
        logging.info("tracking top No.{} in {}".format(num_tracks, vid_proto['video']))
        keep = apply_track_det_nms(tracks, boxes, thres=0.3)
        dets = copy.copy([dets[i] for i in keep])

    track_proto['tracks'] = tracks
    return track_proto


def apply_track_det_nms(tracks, boxes, thres=0.3):
    if len(tracks) == 0:
        return range(len(boxes))
    box_score = np.asarray(boxes, dtype='float32')
    logging.info("Applying nms between tracks ({}) and detections.".format(len(tracks)))
    track_boxes = []
    for track in tracks:
        cur_boxes = [[box['frame'],]+box['bbox'] for box in track]
        track_boxes.extend(cur_boxes)
    track_boxes = np.asarray(track_boxes, dtype='float32')
    keep = track_det_nms(track_boxes, box_score, thres)
    logging.info("{} / {} boxes kept.".format(len(keep), len(boxes)))
    return keep

