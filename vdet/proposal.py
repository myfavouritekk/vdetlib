#!/usr/bin/env python
import os
import numpy as np
import scipy.io
from ..utils.protocol import proto_load, boxes_proto_from_boxes
from ..utils.common import matlab_command, temp_file

def get_windows(image_fnames, cmd='selective_search'):
    """
    Modified from https://github.com/sergeyk/selective_search_ijcv_with_python
    Run MATLAB Selective Search code on the given image filenames to
    generate window proposals.
    Parameters
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

    output_filename = temp_file(suffix='.mat')

    image_fnames = [os.path.abspath(str(name)) for name in image_fnames]
    matlab_command(script, image_fnames, output_filename)

    # swap x-axis and y-axis
    all_boxes = list(scipy.io.loadmat(output_filename)['all_boxes'][0])
    all_boxes = [boxes[:, [1,0,3,2]] for boxes in all_boxes]

    # Remove temporary file, and return.
    os.remove(output_filename)
    if len(all_boxes) != len(image_fnames):
        raise Exception("Something went wrong computing the windows!")
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

