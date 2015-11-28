#!/usr/bin/env python

import argparse
import os
import sys
sys.path.insert(1, os.path.join(os.path.dirname(__file__), '../..'))
from vdetlib.utils.common import stem
from vdetlib.utils.protocol import vid_proto_from_dir, proto_dump

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('vid_name')
    parser.add_argument('root_dir')
    parser.add_argument('out_file')
    args = parser.parse_args()

    if os.path.isfile(args.out_file):
        print "{} already exists.".format(args.out_file)
        sys.exit(0)
    vid = vid_proto_from_dir(args.root_dir, args.vid_name)
    save_dir = os.path.dirname(args.out_file)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    proto_dump(vid, args.out_file)
