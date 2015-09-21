# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import numpy as np
cimport numpy as np

cdef inline np.float32_t max(np.float32_t a, np.float32_t b):
    return a if a >= b else b

cdef inline np.float32_t min(np.float32_t a, np.float32_t b):
    return a if a <= b else b

def nms(np.ndarray[np.float32_t, ndim=2] dets, np.float thresh):
    cdef np.ndarray[np.float32_t, ndim=1] x1 = dets[:, 0]
    cdef np.ndarray[np.float32_t, ndim=1] y1 = dets[:, 1]
    cdef np.ndarray[np.float32_t, ndim=1] x2 = dets[:, 2]
    cdef np.ndarray[np.float32_t, ndim=1] y2 = dets[:, 3]
    cdef np.ndarray[np.float32_t, ndim=1] scores = dets[:, 4]

    cdef np.ndarray[np.float32_t, ndim=1] areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    cdef np.ndarray[np.int_t, ndim=1] order = scores.argsort()[::-1]

    cdef int ndets = dets.shape[0]
    cdef np.ndarray[np.int_t, ndim=1] suppressed = \
            np.zeros((ndets), dtype=np.int)

    # nominal indices
    cdef int _i, _j
    # sorted indices
    cdef int i, j
    # temp variables for box i's (the box currently under consideration)
    cdef np.float32_t ix1, iy1, ix2, iy2, iarea
    # variables for computing overlap with box j (lower scoring box)
    cdef np.float32_t xx1, yy1, xx2, yy2
    cdef np.float32_t w, h
    cdef np.float32_t inter, ovr

    keep = []
    for _i in range(ndets):
        i = order[_i]
        if suppressed[i] == 1:
            continue
        keep.append(i)
        ix1 = x1[i]
        iy1 = y1[i]
        ix2 = x2[i]
        iy2 = y2[i]
        iarea = areas[i]
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            xx1 = max(ix1, x1[j])
            yy1 = max(iy1, y1[j])
            xx2 = min(ix2, x2[j])
            yy2 = min(iy2, y2[j])
            w = max(0.0, xx2 - xx1 + 1)
            h = max(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (iarea + areas[j] - inter)
            if ovr >= thresh:
                suppressed[j] = 1

    return keep


def vid_nms(np.ndarray[np.float32_t, ndim=2] dets, np.float thresh):
    cdef np.ndarray[np.float32_t, ndim=1] f_idx = dets[:, 0]
    cdef np.ndarray[np.float32_t, ndim=1] x1 = dets[:, 1]
    cdef np.ndarray[np.float32_t, ndim=1] y1 = dets[:, 2]
    cdef np.ndarray[np.float32_t, ndim=1] x2 = dets[:, 3]
    cdef np.ndarray[np.float32_t, ndim=1] y2 = dets[:, 4]
    cdef np.ndarray[np.float32_t, ndim=1] scores = dets[:, 5]

    cdef np.ndarray[np.float32_t, ndim=1] areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    cdef np.ndarray[np.int_t, ndim=1] order = scores.argsort()[::-1]

    cdef int ndets = dets.shape[0]
    cdef np.ndarray[np.int_t, ndim=1] suppressed = \
            np.zeros((ndets), dtype=np.int)

    # nominal indices
    cdef int _i, _j
    # sorted indices
    cdef int i, j
    # temp variables for box i's (the box currently under consideration)
    cdef np.float32_t ix1, iy1, ix2, iy2, iarea
    # variables for computing overlap with box j (lower scoring box)
    cdef np.float32_t xx1, yy1, xx2, yy2
    cdef np.float32_t w, h
    cdef np.float32_t inter, ovr

    keep = []
    for _i in range(ndets):
        i = order[_i]
        if suppressed[i] == 1:
            continue
        keep.append(i)
        ix1 = x1[i]
        iy1 = y1[i]
        ix2 = x2[i]
        iy2 = y2[i]
        iarea = areas[i]
        for _j in range(_i + 1, ndets):
            j = order[_j]
            # on different frames, no suppression
            if f_idx[i] != f_idx[j]:
                continue
            if suppressed[j] == 1:
                continue
            xx1 = max(ix1, x1[j])
            yy1 = max(iy1, y1[j])
            xx2 = min(ix2, x2[j])
            yy2 = min(iy2, y2[j])
            w = max(0.0, xx2 - xx1 + 1)
            h = max(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (iarea + areas[j] - inter)
            if ovr >= thresh:
                suppressed[j] = 1
    return keep


def track_det_nms(np.ndarray[np.float32_t, ndim=2] tracks,
                  np.ndarray[np.float32_t, ndim=2] dets,
                  np.float thresh):
    # track coordinates
    cdef np.ndarray[np.float32_t, ndim=1] t_f_idx = tracks[:, 0]
    cdef np.ndarray[np.float32_t, ndim=1] t_x1 = tracks[:, 1]
    cdef np.ndarray[np.float32_t, ndim=1] t_y1 = tracks[:, 2]
    cdef np.ndarray[np.float32_t, ndim=1] t_x2 = tracks[:, 3]
    cdef np.ndarray[np.float32_t, ndim=1] t_y2 = tracks[:, 4]
    cdef np.ndarray[np.float32_t, ndim=1] t_areas = (t_x2 - t_x1 + 1) * (t_y2 - t_y1 + 1)

    # det coordinates
    cdef np.ndarray[np.float32_t, ndim=1] f_idx = dets[:, 0]
    cdef np.ndarray[np.float32_t, ndim=1] x1 = dets[:, 1]
    cdef np.ndarray[np.float32_t, ndim=1] y1 = dets[:, 2]
    cdef np.ndarray[np.float32_t, ndim=1] x2 = dets[:, 3]
    cdef np.ndarray[np.float32_t, ndim=1] y2 = dets[:, 4]
    cdef np.ndarray[np.float32_t, ndim=1] scores = dets[:, 5]
    cdef np.ndarray[np.float32_t, ndim=1] areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    cdef int ndets = dets.shape[0]
    cdef int ntracks = tracks.shape[0]
    cdef np.ndarray[np.int_t, ndim=1] suppressed = \
            np.zeros((ndets), dtype=np.int)

    # indices for dets and tracks
    cdef int i, j
    # temp variables for box i's (the box currently under consideration)
    cdef np.float32_t ix1, iy1, ix2, iy2, iarea
    # variables for computing overlap with track j (lower scoring box)
    cdef np.float32_t xx1, yy1, xx2, yy2
    cdef np.float32_t w, h
    cdef np.float32_t inter, ovr

    # first round suppression using tracks
    for i in range(ndets):
        ix1 = x1[i]
        iy1 = y1[i]
        ix2 = x2[i]
        iy2 = y2[i]
        iarea = areas[i]
        for j in range(ntracks):
            if f_idx[i] != t_f_idx[j]:
                # do not suppress across frames
                continue
            xx1 = max(ix1, t_x1[j])
            yy1 = max(iy1, t_y1[j])
            xx2 = min(ix2, t_x2[j])
            yy2 = min(iy2, t_y2[j])
            w = max(0.0, xx2 - xx1 + 1)
            h = max(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (iarea + t_areas[j] - inter)
            if ovr >= thresh:
                suppressed[i] = 1
                break

    # second round: vid_nms among remaining boxes
    cdef np.ndarray[np.int_t, ndim=1] remain = np.where(suppressed == 0)[0]
    keep = vid_nms(dets[remain,:], thresh)

    return [remain[i] for i in keep]

