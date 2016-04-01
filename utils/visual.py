#!/usr/bin/env python
import cv2
import copy
import colorsys
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def unique_colors(N):
    HSV_tuples = [(x*1.0/N, 1, 0.8) for x in range(N)]
    return map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)

def add_bbox(img, boxes, colors=None, line_width=2):
    result = copy.copy(img)
    if colors is None:
        colors = unique_colors(len(boxes))
    for box_id, bbox in enumerate(boxes):
        if bbox is None:
            continue
        bbox = map(int, bbox)
        color = colors[box_id]
        color = tuple([int(255*x) for x in color])
        cv2.rectangle(result, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                color, line_width)
    return result

def plot(x, ys, labels=None, colors=None, line_width=5):
    if colors is None:
        colors = unique_colors(len(ys))
    if labels is None:
        labels = [None] * len(ys)
    fig = plt.figure()
    for y, color, label in zip(ys, colors, labels):
        plt.plot(x, y, color=color, label=label, lw=line_width)
    plt.legend(loc=4)
    return fig

def plot_track_scores(score_proto, legend=True):
    import numpy as np
    plots = []
    for tubelet in score_proto['tubelets']:
        track = {}
        track['length'] = len(tubelet['boxes'])
        if track['length'] <= 0:
            continue
        fields = tubelet['boxes'][0].keys()
        track['det_scores'] = map(lambda x:x['det_score'],
                                  tubelet['boxes'])
        track['anchors'] = map(lambda x:x['anchor'] * 1. / track['length'],
                                  tubelet['boxes'])
        track['abs_anchors'] = map(abs, track['anchors'])
        if 'gt_overlap' in fields:
            track['gt_overlaps'] = map(lambda x:x['gt_overlap'],
                                  tubelet['boxes'])
        if 'conv_score' in fields:
            track['conv_scores'] = map(lambda x:x['conv_score'],
                                  tubelet['boxes'])
        if 'track_score' in fields:
            track['track_scores'] = map(lambda x:x['track_score'],
                                  tubelet['boxes'])
        track['frames'] = map(lambda x:x['frame'],
                                  tubelet['boxes'])
        x = track['frames']
        y = []
        if legend:
            labels = []
            for label in set(['det_scores', 'track_scores', 'anchors', 'gt_overlaps',
                          'conv_scores']).intersection(set(track.keys())):
                y.append(track[label])
                labels.append(label)
            plots.append(plot(x, y, labels))
        else:
            plots.append(plot(x, y, None))
    return plots