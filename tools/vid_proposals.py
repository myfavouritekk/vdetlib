#!/usr/bin/env python
import argparse
import sys
import cv2
sys.path.insert(1, '.')
from vdetlib.vdet.proposal import vid_proposals
from vdetlib.utils.protocol import proto_load, proto_dump

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('vid_file')
    parser.add_argument('save_file')
    args = parser.parse_args()

    vid_proto = proto_load(args.vid_file)
    box_proto = vid_proposals(vid_proto)
    proto_dump(box_proto, args.save_file)
