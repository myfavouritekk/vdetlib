#!/usr/bin/env python
import os
from scipy.io import loadmat
import numpy as np
from ..utils.protocol import frame_path_after, frame_path_before, tracks_proto_from_boxes
from ..utils.common import matlab_command, temp_file


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
        print "Forward tracking failed."
        fw_trk = [bbox+[1.]]+[[float('nan')]*5]*(len(fw_frames)-1)

    try:
        bw_trk = loadmat(bw_out)['bbox']
    except:
        print "Backward tracking failed."
        bw_trk = [[float('nan')]*5]*(len(bw_frames)-1) + [bbox+[1.]]

    os.remove(fw_out)
    os.remove(bw_out)
    bw_trk = bw_trk[::-1]
    trk = np.concatenate((bw_trk, fw_trk[1:]))
    tracks_proto = tracks_proto_from_boxes(trk, vid_proto['video'])
    return tracks_proto


def track_from_det(vid_proto, det_proto, track_method):
    assert vid_proto['video'] == det_proto['video']
    track_proto = {}
    track_proto['video'] = vid_proto['video']
    track_proto['method'] = track_method.__name__
    tracks = []
    for idx, det in enumerate(det_proto['detections'], start=1):
        print "tracking top No.{} in {}".format(idx, vid_proto['video'])
        tracks.extend(track_method(vid_proto, det))
    track_proto['tracks'] = tracks
    return track_proto

