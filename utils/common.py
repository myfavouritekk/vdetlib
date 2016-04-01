import os
import re
import argparse
import codecs
import cPickle
import numpy as np
import matplotlib as mpl
from scipy.misc import imresize
from multiprocessing import Pool
import scipy.io as sio
import cv2
import subprocess
import shlex
from log import logging
from easydict import EasyDict as edict

def pickle(data, file_path):
    with open(file_path, 'wb') as f:
        cPickle.dump(data, f, cPickle.HIGHEST_PROTOCOL)


def unpickle(file_path):
    with open(file_path, 'rb') as f:
        data = cPickle.load(f)
    return data


def read_list(file_path, coding=None):
    if coding is None:
        with open(file_path, 'r') as f:
            arr = [line.strip() for line in f.readlines()]
    else:
        with codecs.open(file_path, 'r', coding) as f:
            arr = [line.strip() for line in f.readlines()]
    return arr


def write_list(arr, file_path, coding=None):
    if coding is None:
        arr = ['{}'.format(item) for item in arr]
        with open(file_path, 'w') as f:
            f.write('\n'.join(arr))
    else:
        with codecs.open(file_path, 'w', coding) as f:
            f.write(u'\n'.join(arr))


def read_window_file(file_path):
    arr = read_list(file_path)
    ret = []
    i = 0
    while i < len(arr):
        if len(ret) != int(arr[i][2:]):
            raise ValueError(
                "Window file {}: bad index at line {}".format(file_path, i))
        obj = {
            'img_path': arr[i + 1],
            'channels': int(arr[i + 2]),
            'height': int(arr[i + 3]),
            'width': int(arr[i + 4]),
            'windows': []
        }
        num = int(arr[i + 5])
        i += 5
        while num > 0:
            i += 1
            tmp = arr[i].split()
            label = int(tmp[0])
            overlap = float(tmp[1])
            bbox = np.asarray(map(int, tmp[2:]))
            obj['windows'].append((label, overlap, bbox))
            num -= 1
        i += 1
        ret.append(obj)
    return ret


def read_window_seg_file(file_path):
    arr = read_list(file_path)
    ret = []
    i = 0
    while i < len(arr):
        if len(ret) != int(arr[i][2:]):
            raise ValueError(
                "Window seg file {}: bad index at line {}".format(file_path, i))
        obj = {
            'img_path': arr[i + 1],
            'gt_path': arr[i + 2],
            'channels': int(arr[i + 3]),
            'height': int(arr[i + 4]),
            'width': int(arr[i + 5]),
            'windows': []
        }
        num = int(arr[i + 6])
        i += 6
        while num > 0:
            i += 1
            tmp = arr[i].split()
            label = int(tmp[0])
            overlap = float(tmp[1])
            bbox = np.asarray(map(int, tmp[2:]))
            obj['windows'].append((label, overlap, bbox))
            num -= 1
        i += 1
        ret.append(obj)
    return ret


def write_window_file(file_path, windows):
    lines = []
    for i, obj in enumerate(windows):
        lines.append('# {}'.format(i))
        lines.append(obj['img_path'])
        lines.append(str(obj['channels']))
        lines.append(str(obj['height']))
        lines.append(str(obj['width']))
        lines.append(str(len(obj['windows'])))
        for label, overlap, bbox in obj['windows']:
            bbox = ' '.join(map(str, bbox))
            lines.append('{} {:.3f} {}'.format(label, overlap, bbox))
    write_list(lines, file_path)

def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)


def img_crop(img, in_bbox, crop_mode, crop_size, padding, pad_value, gt=None):
    bbox = in_bbox - 1
    use_square = True if crop_mode == 'square' else False
    pad_w = 0
    pad_h = 0;
    crop_width = crop_size
    crop_height = crop_size
    if padding > 0 or use_square:
        scale = crop_size * 1.0 / (crop_size - padding * 2)
        half_height = (bbox[3] - bbox[1] + 1) / 2.0
        half_width = (bbox[2] - bbox[0] + 1) / 2.0
        center = [bbox[0] + half_width, bbox[1] + half_height]
        if use_square:
            if half_height > half_width:
                half_width = half_height
            else:
                half_height = half_width
        bbox = [center[0] - half_width * scale,
                center[1] - half_height * scale,
                center[0] + half_width * scale,
                center[1] + half_height * scale]
        unclipped_bbox = map(int, map(round, bbox))
        unclipped_height = bbox[3] - bbox[1] + 1
        unclipped_width = bbox[2] - bbox[0] + 1
        pad_x1 = max(0, -bbox[0])
        pad_y1 = max(0, -bbox[1])
        bbox[0] = max(0, bbox[0])
        bbox[1] = max(0, bbox[1])
        bbox[2] = min(img.shape[1] - 1, bbox[2])
        bbox[3] = min(img.shape[0] - 1, bbox[3])
        clipped_height = bbox[3] - bbox[1] + 1
        clipped_width = bbox[2] - bbox[0] + 1
        scale_x = crop_size * 1.0 / unclipped_width;
        scale_y = crop_size * 1.0 / unclipped_height
        crop_width = int(round(clipped_width * scale_x))
        crop_height = int(round(clipped_height * scale_y))
        pad_x1 = int(round(pad_x1 * scale_x))
        pad_y1 = int(round(pad_y1 * scale_y))
        pad_h = pad_y1
        pad_w = pad_x1
        if pad_y1 + crop_height > crop_size:
            crop_height = crop_size - pad_y1
        if pad_x1 + crop_width > crop_size:
            crop_width = crop_size - pad_x1

    img_window = img[bbox[1]:bbox[3]+1, bbox[0]:bbox[2]+1, :]
    tmp = imresize(img_window, [crop_height, crop_width])
    img_window = np.ones((crop_size, crop_size, 3), np.uint8) * \
        pad_value.reshape((1,1,3)).astype(np.uint8)
    img_window[pad_h:pad_h+crop_height, pad_w:pad_w+crop_width] = tmp
    # img_window = (img_window / 255.0).astype(np.float32)

    visual_debug = False
    if visual_debug:
        cv2.imshow('proposal', img_window)
        cv2.waitKey(0)

    if gt is None:
        return img_window, unclipped_bbox

    gt_window = gt[bbox[1]:bbox[3]+1, bbox[0]:bbox[2]+1]
    tmp = imresize(gt_window, [crop_height, crop_width], interp='nearest')
    gt_window = np.zeros((crop_size, crop_size), np.uint8)
    gt_window[pad_h:pad_h+crop_height, pad_w:pad_w+crop_width] = tmp
    return img_window, gt_window, unclipped_bbox


def rcnn_img_crop(img, in_bbox, crop_mode, crop_size, padding, image_mean=None):
    bbox = in_bbox - 1
    use_square = True if crop_mode == 'square' else False
    pad_w = 0
    pad_h = 0;
    crop_width = crop_size
    crop_height = crop_size
    if padding > 0 or use_square:
        scale = crop_size * 1.0 / (crop_size - padding * 2)
        half_height = (bbox[3] - bbox[1] + 1) / 2.0
        half_width = (bbox[2] - bbox[0] + 1) / 2.0
        center = [bbox[0] + half_width, bbox[1] + half_height]
        if use_square:
            if half_height > half_width:
                half_width = half_height
            else:
                half_height = half_width
        bbox = map(round, [center[0] - half_width * scale,
                center[1] - half_height * scale,
                center[0] + half_width * scale,
                center[1] + half_height * scale])
        unclipped_height = bbox[3] - bbox[1] + 1
        unclipped_width = bbox[2] - bbox[0] + 1
        pad_x1 = max(0, -bbox[0])
        pad_y1 = max(0, -bbox[1])
        bbox[0] = max(0, bbox[0])
        bbox[1] = max(0, bbox[1])
        bbox[2] = min(img.shape[1] - 1, bbox[2])
        bbox[3] = min(img.shape[0] - 1, bbox[3])
        clipped_height = bbox[3] - bbox[1] + 1
        clipped_width = bbox[2] - bbox[0] + 1
        scale_x = crop_size * 1.0 / unclipped_width;
        scale_y = crop_size * 1.0 / unclipped_height
        crop_width = int(round(clipped_width * scale_x))
        crop_height = int(round(clipped_height * scale_y))
        pad_x1 = int(round(pad_x1 * scale_x))
        pad_y1 = int(round(pad_y1 * scale_y))
        pad_h = pad_y1
        pad_w = pad_x1
        if pad_y1 + crop_height > crop_size:
            crop_height = crop_size - pad_y1
        if pad_x1 + crop_width > crop_size:
            crop_width = crop_size - pad_x1

    bbox = map(int, bbox)
    img_window = img[bbox[1]:bbox[3]+1, bbox[0]:bbox[2]+1, :]
    tmp = cv2.resize(img_window.astype('float'), (crop_width, crop_height),
            interpolation=cv2.INTER_LINEAR).astype('float')
    if image_mean is not None:
        tmp -= image_mean.reshape((1,1,3))
    img_window = np.zeros((crop_size, crop_size, 3)).astype('single')
    img_window[pad_h:pad_h+crop_height, pad_w:pad_w+crop_width] = tmp

    visual_debug = False
    if visual_debug:
        cv2.imshow('proposal', img_window / 255.)
        cv2.waitKey(0)

    return img_window

def im_transform(image, size=-1, scale=1., mean_values=[0.,0.,0.]):
    assert len(mean_values) == 3
    if size > 0:
        try:
            patch = cv2.resize(image, (size, size))
        except:
            logging.error("Unable to resize image.")
            print image.shape
            raise
    else:
        patch = image
    trans_img = (np.asarray(patch) - np.asarray(mean_values)) * scale
    return trans_img.swapaxes(1,2).swapaxes(0,1)

def get_voc_cmap():
    cmap = np.zeros((256, 3))
    for i in xrange(256):
        k = i
        r = g = b = 0
        for j in xrange(7):
            r = (r | ((k & 1) << (7 - j)))
            g = (g | (((k & 2) >> 1) << (7 - j)))
            b = (b | (((k & 4) >> 2) << (7 - j)))
            k = (k >> 3)
        cmap[i] = [r, g, b]
    cmap = cmap / 255.0
    return mpl.colors.ListedColormap(cmap, name='voc_cmap')


def os_command(command):
    import subprocess32 as sp
    sp.call(command)


def matlab_command(fun_file, input_list, outfile):
    '''matlab function wrapper, matlab
        function should take two arguments,
        fun(infile, outfile)
    '''
    script_dirname = os.path.abspath(os.path.dirname(fun_file))
    cmd = stem(fun_file)
    fnames_cell = '{' + ','.join(["{}".format(x) if type(x) is not str \
        else "'{}'".format(x) for x in input_list]) + '}'
    command = "{}({}, '{}')".format(cmd, fnames_cell, outfile)

    # Execute command in MATLAB.
    debug = False
    if debug:
        print(command)
        mc = "matlab -nojvm -r \"{}; exit\"".format(command)
        pid = subprocess.Popen(shlex.split(mc), cwd=script_dirname)
    else:
        mc = "matlab -nodisplay -nojvm -nosplash -nodesktop \
              -r \"try; {}; catch; exit; end; exit\"".format(command)
        pid = subprocess.Popen(
            shlex.split(mc), stdout=open('/dev/null', 'w'), cwd=script_dirname)
    retcode = pid.wait()
    if retcode != 0:
        raise Exception("Matlab script did not exit successfully!")
    else:
        return True


def matlab_engine(fun_file, input_list, eng=None):
    '''matlab enginer wrapper
    return_val = fun(input_list)
    '''
    import matlab.engine
    script_dirname = os.path.abspath(os.path.dirname(fun_file))
    fun_name = stem(fun_file)
    if eng is None:
        eng = matlab.engine.start_matlab('-nodisplay -nojvm -nosplash -nodesktop')
    else:
        logging.debug("Use opened Matlab session: {}".format(eng))
    try:
        eng.cd(script_dirname)
        func = getattr(eng, fun_name)
        result = func(input_list)
    except matlab.engine.EngineError, e:
        # Use new engine instead
        logging.error("Existing engine no response: {}".format(eng))
        raise e
    if eng is None:
        logging.debug("before matlab quiting...")
        eng.quit()
        logging.debug("after matlab quiting...")
    eng.clear(nargout=0)
    return result


def parallel_map(pool, func, input_list):
    return pool.map(func, input_list)


def basename(file_path):
    return os.path.basename(file_path)


def stem(file_path):
    return os.path.splitext(basename(file_path))[0]


def isimg(name):
    exts = ['.jpeg', '.png', '.jpg']
    return any([name.lower().endswith(ext) for ext in exts])


def imread(image_path):
    return cv2.imread(image_path, cv2.IMREAD_COLOR)


def imwrite(filename, img):
    cv2.imwrite(filename, img)


def caffe_net(model, param, gpu_id=0, phase='test'):
    import sys
    sys.path.insert(1, os.path.join(os.path.dirname(__file__),
                        '../../External/caffe/python/'))
    import caffe
    if phase is 'test':
        phase = caffe.TEST
    else:
        phase = caffe.TRAIN
    os.environ['GLOG_minloglevel'] = '2'
    caffe.set_device(gpu_id)
    caffe.set_mode_gpu()
    net = caffe.Net(model, param, phase)
    return net


def temp_file(suffix=''):
    import tempfile
    f, output_filename = tempfile.mkstemp(suffix=suffix)
    os.close(f)
    return output_filename


def quick_args(arglist):
    parser = argparse.ArgumentParser()
    for arg in arglist:
        if type(arg) == tuple:
            parser.add_argument(arg[0], type=arg[1])
        else:
            parser.add_argument(arg)
    return parser.parse_args()


def svm_from_rcnn_model(rcnn_model):
    rcnn_model = sio.loadmat(rcnn_model)['rcnn_model']
    detectors = rcnn_model['detectors'][0,0]
    svm = {}
    svm['W'] = detectors['W'][0,0]
    svm['B'] = detectors['B'][0,0]
    svm['feat_norm_mean'] = rcnn_model['training_opts'][0,0]['feat_norm_mean'][0,0][0,0]
    return svm

import errno
from functools import wraps

class TimeoutError(Exception):
    pass

def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        def wrapper(*args, **kwargs):
            import signal
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wraps(func)(wrapper)

    return decorator


def iou(boxes1, boxes2):
    boxes1 = np.asarray(boxes1).astype('float')
    boxes2 = np.asarray(boxes2).astype('float')
    # intersection boundaries, widths and heights
    ix1 = np.maximum(boxes1[:,[0]], boxes2[:,[0]].T)
    ix2 = np.minimum(boxes1[:,[2]], boxes2[:,[2]].T)
    iy1 = np.maximum(boxes1[:,[1]], boxes2[:,[1]].T)
    iy2 = np.minimum(boxes1[:,[3]], boxes2[:,[3]].T)
    iw = np.maximum(0, ix2 - ix1 + 1)
    ih = np.maximum(0, iy2 - iy1 + 1)
    # areas
    areas1 = (boxes1[:, [2]] - boxes1[:, [0]] + 1) * \
             (boxes1[:, [3]] - boxes1[:, [1]] + 1)
    areas2 = (boxes2[:, [2]] - boxes2[:, [0]] + 1) * \
             (boxes2[:, [3]] - boxes2[:, [1]] + 1)
    inter = iw * ih
    overlaps = 1. * inter / (areas1 + areas2.T - inter)
    return overlaps

def options(option_dict):
    return edict(option_dict)
