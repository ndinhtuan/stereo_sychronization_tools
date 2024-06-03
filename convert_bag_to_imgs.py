#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2016 Massachusetts Institute of Technology

"""Extract images from a rosbag.
"""

import os
import argparse

import cv2

import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

def main():
    """Extract a folder of images from a rosbag.
    """
    parser = argparse.ArgumentParser(description="Extract images from a ROS bag.")
    parser.add_argument("bag_file", help="Input ROS bag.")
    parser.add_argument("--left_output_dir", help="Left output directory.", default="/media/tuan/Daten/mapathon/ikg_cam_exterior_calib/3/left")
    parser.add_argument("--right_output_dir", help="Right output directory.", default="/media/tuan/Daten/mapathon/ikg_cam_exterior_calib/3/right")
    parser.add_argument("--left_image_topic", help="Left image topic.", default="/stereo/left/image_color")
    parser.add_argument("--right_image_topic", help="Right image topic.", default="/stereo/right/image_color")

    args = parser.parse_args()

    # print("Extract images from %s on topic %s into %s" % (args.bag_file,
    #                                                       args.image_topic, args.output_dir))

    bag = rosbag.Bag(args.bag_file, "r")
    bridge = CvBridge()
    count = 0
    for (ltopic, lmsg, lt)  in bag.read_messages(topics=[args.left_image_topic]):
        # Saving left image
        cv_img = bridge.imgmsg_to_cv2(lmsg, desired_encoding="passthrough")

        cv2.imwrite(os.path.join(args.left_output_dir, "frame%06i.png" % count), cv_img)
        print("Wrote image {}".format(count))
        count += 1

    count = 0
    for (rtopic, rmsg, rt)  in bag.read_messages(topics=[args.right_image_topic]):
        # Saving left image
        cv_img = bridge.imgmsg_to_cv2(rmsg, desired_encoding="passthrough")

        cv2.imwrite(os.path.join(args.right_output_dir, "frame%06i.png" % count), cv_img)
        print("Wrote image {}".format(count))
        count += 1

    bag.close()

    return

if __name__ == '__main__':
    main()
