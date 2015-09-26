#!/usr/bin/env python
import os
import numpy as np
import scipy.io
from ..utils.protocol import proto_load, boxes_proto_from_boxes
from ..utils.common import matlab_engine, temp_file

def get_windows(image_fnames, cmd='selective_search'):
    """
    ----------
    image_filenames: strings
        Paths to images to run on.
    cmd: string
        selective search function to call:
            - 'selective_search' for a few quick proposals
            - 'selective_seach_rcnn' for R-CNN configuration for more coverage.
    """
    # Form the MATLAB script command that processes images and write to
    # temporary results file.
    assert cmd in ['selective_search', 'selective_seach_rcnn']
    script = os.path.join(os.path.dirname(__file__),
        '../../External/selective_search_python/{}.m'.format(cmd))

    image_fnames = [os.path.abspath(str(name)) for name in image_fnames]
    all_boxes = matlab_engine(script, image_fnames)

    # swap x-axis and y-axis
    all_boxes = [np.asarray(boxes, dtype='int')[:, [1,0,3,2]] \
                    for boxes in all_boxes]
    return all_boxes


def vid_proposals(vid_proto, method='selective_search'):
    box_proto = {}
    box_proto['video'] = vid_proto['video']
    frame_names = [os.path.join(vid_proto['root_path'], x['path']) \
                    for x in vid_proto['frames']]
    frame_idx = [x['frame'] for x in vid_proto['frames']]
    proposals = get_windows(frame_names, method)
    box_proto['boxes'] = boxes_proto_from_boxes(frame_idx, proposals,
                                                box_proto['video'])
    return box_proto

