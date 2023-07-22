'''
Sample Command:-
python detect_aruco_images.py --image Images/test_image_1.png --type DICT_5X5_100
'''
import numpy as np
from aruco_utils import ARUCO_DICT, aruco_display, aruco_classification, draw_simulation_led_plate
import argparse
import cv2
import sys
import glob


ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True, help="path to dir containing images")
ap.add_argument("-t", "--type", type=str, default="DICT_4X4_50", help="type of ArUCo tag to detect")
args = vars(ap.parse_args())



for p in glob.glob("{}/*.png".format(args["path"])):

    print("Loading image...")
    image = cv2.imread(p)
    h,w,_ = image.shape
    width=600
    height = int(width*(h/w))
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)


    # verify that the supplied ArUCo tag exists and is supported by OpenCV
    if ARUCO_DICT.get(args["type"], None) is None:
        print(f"ArUCo tag type '{args['type']}' is not supported")
        sys.exit(0)

    # load the ArUCo dictionary, grab the ArUCo parameters, and detect
    # the markers
    print("Detecting '{}' tags....".format(args["type"]))
    arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[args["type"]])
    arucoParams = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)

    detected_markers, gray_plate = aruco_display(detector, image)
    cv2.imshow("Image", detected_markers)

    if gray_plate is not None:
        tmp = aruco_classification(gray_plate, 100)
        sim_plate = draw_simulation_led_plate(tmp, background=None)
        cv2.imshow("sim_plate", sim_plate)


    # # Uncomment to save
    # cv2.imwrite("output_sample.png",detected_markers)

    cv2.waitKey(0)