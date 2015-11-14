#!/usr/bin/env python

import argparse
import json
from xml.dom import minidom
import xmltodict
import os
import glob
import sys

name_map = {
    'n02691156': (1, 'airplane'),
    'n02419796': (2, 'antelope'),
    'n02131653': (3, 'bear'),
    'n02834778': (4, 'bicycle'),
    'n01503061': (5, 'bird'),
    'n02924116': (6, 'bus'),
    'n02958343': (7, 'car'),
    'n02402425': (8, 'cattle'),
    'n02084071': (9, 'dog'),
    'n02121808': (10, 'domestic_cat'),
    'n02503517': (11, 'elephant'),
    'n02118333': (12, 'fox'),
    'n02510455': (13, 'giant_panda'),
    'n02342885': (14, 'hamster'),
    'n02374451': (15, 'horse'),
    'n02129165': (16, 'lion'),
    'n01674464': (17, 'lizard'),
    'n02484322': (18, 'monkey'),
    'n03790512': (19, 'motorcycle'),
    'n02324045': (20, 'rabbit'),
    'n02509815': (21, 'red_panda'),
    'n02411705': (22, 'sheep'),
    'n01726692': (23, 'snake'),
    'n02355227': (24, 'squirrel'),
    'n02129604': (25, 'tiger'),
    'n04468005': (26, 'train'),
    'n01662784': (27, 'turtle'),
    'n04530566': (28, 'watercraft'),
    'n02062744': (29, 'whale'),
    'n02391049': (30, 'zebra')
}


def track_by_id(annotations, track_id):
    tracks = [track for track in annotations if track['id'] == track_id]
    assert len(tracks) <= 1
    if tracks:
        return tracks[0]
    else:
        return []

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('annot_dir')
    parser.add_argument('save_file')
    args = parser.parse_args()

    if os.path.isfile(args.save_file):
        print "{} already exists.".format(args.save_file)
        sys.exit(0)

    vid_name = os.path.basename(args.annot_dir)
    annot = {}
    annot['video'] = vid_name
    annotations = []
    xml_list = glob.glob(os.path.join(args.annot_dir, '*.xml'))
    for xml_file in xml_list:
        with open(xml_file, 'r') as f:
            xml = xmltodict.parse(f.read())['annotation']
        frame = int(xml['filename']) + 1 # frame id
        frame_width = int(xml['size']['width'])
        frame_height = int(xml['size']['height'])
        try:
            obj = xml['object']
        except KeyError, e:
            print "xml {} has no objects.".format(xml_file)
            continue
        if type(obj) is not list:
            boxes = [obj]
        else:
            boxes = obj
        for box in boxes:
            track_id = str(box['trackid'])
            track = track_by_id(annotations, track_id)
            if not track:
                track = {
                    "id": track_id,
                    "track": []
                }
                annotations.append(track)
            bbox = map(int, [box['bndbox']['xmin'],
                             box['bndbox']['ymin'],
                             box['bndbox']['xmax'],
                             box['bndbox']['ymax']])
            name = str(box['name'])
            cls_name = name_map[name][1]
            cls_idx = name_map[name][0]
            generated = int(box['generated'])
            occluded = int(box['occluded'])
            track['track'].append(
                    {
                        "frame": frame,
                        "bbox": bbox,
                        "name": name,
                        "class": cls_name,
                        "class_index": cls_idx,
                        "generated": generated,
                        "occluded": occluded,
                        "frame_size": [frame_height, frame_width]
                    }
                )

    annot['annotations'] = annotations
    if not os.path.isdir(os.path.dirname(args.save_file)):
        os.makedirs(os.path.dirname(args.save_file))
    with open(args.save_file, 'w') as f:
        json.dump(annot, f, indent=2)
