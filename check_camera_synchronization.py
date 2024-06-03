import cv2
import numpy as np 
import argparse
import glob
from detect_aruco import get_gray_code
from aruco_utils import ARUCO_DICT, aruco_display, aruco_classification, draw_simulation_led_plate
import sys
import os

ap = argparse.ArgumentParser()
ap.add_argument("-lp", "--left_path", required=True, help="path to dir containing left images")
ap.add_argument("-rp", "--right_path", required=True, help="path to dir containing right images")
ap.add_argument("-t", "--type", type=str, default="DICT_4X4_50", help="type of ArUCo tag to detect")
args = vars(ap.parse_args())

def read_image(path):

    image = cv2.imread(path)
    h,w,_ = image.shape
    width=600
    height = int(width*(h/w))
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)
    return image

if __name__=="__main__":

    path_to_result = "./result"
    path_to_result_synch = "{}/synch".format(path_to_result)
    path_to_result_notsynch = "{}/notsynch".format(path_to_result)
    path_to_result_notfound = "{}/notfound".format(path_to_result)

    os.makedirs(path_to_result, exist_ok=False)
    os.makedirs(path_to_result_synch, exist_ok=False)
    os.makedirs(path_to_result_notsynch, exist_ok=False)
    os.makedirs(path_to_result_notfound, exist_ok=False)

    list_left_img = glob.glob("{}/*.png".format(args["left_path"]))
    list_right_img = glob.glob("{}/*.png".format(args["right_path"]))

    assert len(list_left_img) == len(list_right_img), " number of left images should equal to number of right images"

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

    file_ = open("{}/myfile.txt".format(path_to_result),"w")
    count_notfound = 0
    count_synch = 0
    count_nosynch = 0

    for limg_path in list_left_img:

        name_img = limg_path.split('/')[-1]
        rimg_path = os.path.join(args["right_path"], name_img)

        limg = read_image(limg_path)
        rimg = read_image(rimg_path)

        ltime = get_gray_code(detector, limg, False)
        rtime = get_gray_code(detector, rimg, False)

        re = "{} Don't see gray code in both left and right images \n".format(name_img)
        img_show = cv2.hconcat([limg, rimg])

        status = 0
        if ltime is not None and rtime is not None:
            
            if ltime == rtime:
                re = "{} Left and right camera is synchornize \n".format(name_img)
                status = 1
            else:
                status = 2
                re = "{} Left and Right is not synch, with timestamp: L-{} and R-{} \n".format(name_img, ltime, rtime)

        status_str = ""

        if status == 0:
            count_notfound += 1
            status_str = "not found enough aruco"
            cv2.putText(img_show, "L-R {}".format(status_str),(10, 15), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 0), 2)
            cv2.imwrite("{}/{}".format(path_to_result_notfound, name_img), img_show)
        elif status==1:
            count_synch += 1
            status_str = "synch"
            cv2.putText(img_show, "L-R {}".format(status_str),(10, 15), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 0), 2)
            cv2.imwrite("{}/{}".format(path_to_result_synch, name_img), img_show)
        elif status==2:
            count_nosynch += 1
            status_str = "not synch"
            cv2.putText(img_show, "L-R {}".format(status_str),(10, 15), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 0), 2)
            cv2.imwrite("{}/{}".format(path_to_result_notsynch, name_img), img_show)

        # cv2.imshow("Visualize synch", img_show); cv2.waitKey(0)
        file_.write(re)
    file_.write("Summary: {} pair not found enough aruco, {} pair not synch and {} pair is synch".format(count_notfound, count_nosynch, count_synch))
    
    file_.close()