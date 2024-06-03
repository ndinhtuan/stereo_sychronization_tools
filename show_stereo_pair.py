import cv2 
import glob 
import os

if __name__=="__main__":
    root_path = "/media/tuan/Daten/mapathon/calibration_image_30042024/ife/stereo_calibration_image/processed_data/17082023_dayafter/bag_merged"
    left_camera_folder = "left_camera"
    right_camera_folder = "right_camera"

    list_images = glob.glob("{}/*.png".format(os.path.join(root_path, left_camera_folder)))
    print(list_images)

    for image_path in list_images:
        image_name = image_path.split("/")[-1]
        image_left = cv2.imread(image_path)
        image_right = cv2.imread("{}/{}".format(os.path.join(root_path, right_camera_folder), image_name))

        cv2.imshow("left", image_left)
        cv2.imshow("right", image_right)

        print(image_name)

        c = cv2.waitKey(0)

        if c == 113: # q
            exit()
