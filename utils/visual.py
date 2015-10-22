#!/usr/bin/env python
import cv2
import copy
import colorsys
import matplotlib.pyplot as plt

def unique_colors(N):
    HSV_tuples = [(x*1.0/N, 1, 0.8) for x in range(N)]
    return map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)

def add_bbox(img, boxes, colors=None):
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
                color, 2)
    return result
