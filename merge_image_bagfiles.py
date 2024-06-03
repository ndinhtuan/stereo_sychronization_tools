import numpy as np 
import glob 
import os
import shutil

if __name__=="__main__":
    
    image_bag_folders = ["/media/tuan/Daten/mapathon/calibration_image_30042024/ife/stereo_calibration_image/processed_data/17082023_dayafter/bag_0",
                         "/media/tuan/Daten/mapathon/calibration_image_30042024/ife/stereo_calibration_image/processed_data/17082023_dayafter/bag_1"]
    output_folder = "/media/tuan/Daten/mapathon/calibration_image_30042024/ife/stereo_calibration_image/processed_data/17082023_dayafter/bag_merged"
    left_output_folder = os.path.join(output_folder, "left_camera")
    mono_left_output_folder = os.path.join(left_output_folder, "mono_calibration")

    right_output_folder = os.path.join(output_folder, "right_camera")
    mono_right_output_folder = os.path.join(right_output_folder, "mono_calibration")

    mono_left_count = 0
    mono_right_count = 0
    stereo_count = 0

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        os.makedirs(left_output_folder)
        os.makedirs(mono_left_output_folder)
        os.makedirs(right_output_folder)
        os.makedirs(mono_right_output_folder)
    else:
        print("Merged foler is expected not exists. Please check.")
        exit()

    for image_bag_folder in image_bag_folders:
        # Working with mono image
        ## left
        mono_left_image_bag_folder = os.path.join(image_bag_folder, "left_camera", "mono_calibration")
        list_img = glob.glob("{}/*.png".format(mono_left_image_bag_folder))
        list_id_img = [int(i.split("/")[-1].split(".")[0].split("image")[-1]) for i in list_img]
        max_id_img = max(list_id_img)
        left_max_id_img = max_id_img+1

        for i in range(max_id_img+1):
            shutil.copy(os.path.join(mono_left_image_bag_folder, "image{}.png".format(i)), os.path.join(mono_left_output_folder, "image{}.png".format(mono_left_count)))
            mono_left_count += 1

        ## right
        mono_right_image_bag_folder = os.path.join(image_bag_folder, "right_camera", "mono_calibration")
        list_img = glob.glob("{}/*.png".format(mono_right_image_bag_folder))
        list_id_img = [int(i.split("/")[-1].split(".")[0].split("image")[-1]) for i in list_img]
        max_id_img = max(list_id_img)
        right_max_id_img = max_id_img+1

        for i in range(max_id_img+1):
            shutil.copy(os.path.join(mono_right_image_bag_folder, "image{}.png".format(i)), os.path.join(mono_right_output_folder, "image{}.png".format(mono_right_count)))
            mono_right_count += 1

        # Working with stereo image
        left_image_bag_folder = os.path.join(image_bag_folder, "left_camera")
        right_image_bag_folder = os.path.join(image_bag_folder, "right_camera")

        list_img = glob.glob("{}/*.png".format(left_image_bag_folder))
        list_id_img = [int(i.split("/")[-1].split(".")[0].split("image")[-1]) for i in list_img]
        max_id_img = max(list_id_img)
        stereo_max_id_img = max_id_img+1

        for i in range(max_id_img+1):
            shutil.copy(os.path.join(left_image_bag_folder, "image{}.png".format(i)), os.path.join(left_output_folder, "image{}.png".format(stereo_count)))
            shutil.copy(os.path.join(right_image_bag_folder, "image{}.png".format(i)), os.path.join(right_output_folder, "image{}.png".format(stereo_count)))
            stereo_count += 1
        
        print("Stats for folder {}\n. {} mono left, {} mono right and {} stereo".format(image_bag_folder, left_max_id_img, right_max_id_img, stereo_max_id_img))