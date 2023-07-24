import numpy as np
import cv2
import copy

ARUCO_DICT = {
	"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
	"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
	"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
	"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
	"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
	"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
	"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
	"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
	"DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
	"DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
	"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
	"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

# Hard code coordination for dot on aruco plate
WIDTH_PLATE=100
HEIGHT_PLATE=50
PATTERN_X = [27, 32, 37, 42, 47, 52, 57, 62, 67, 72] # seems step = 5
PATTERN_Y = [16, 23, 31, 38] # seems step = 7
ROW1 = [(i, PATTERN_Y[0]) for t, i in enumerate(PATTERN_X) if t % 2 == 0]
ROW2 = [(i, PATTERN_Y[1]) for t, i in enumerate(PATTERN_X) if t % 2 == 1]
ROW3 = [(i, PATTERN_Y[2]) for t, i in enumerate(PATTERN_X) if t % 2 == 0]
ROW4 = [(i, PATTERN_Y[3]) for t, i in enumerate(PATTERN_X) if t % 2 == 1]

def draw_aruco_box(image, top_left_most, top_right_most, bottom_left_most, bottom_right_most):

    cv2.line(image, top_left_most, top_right_most, (0, 255, 0), 2)
    cv2.line(image, top_right_most, bottom_right_most, (0, 255, 0), 2)
    cv2.line(image, bottom_right_most, bottom_left_most, (0, 255, 0), 2)
    cv2.line(image, bottom_left_most, top_left_most, (0, 255, 0), 2)

def draw_simulation_led_plate(classification_results, background=None):
    
    row1, row2, row3, row4 = classification_results
    if background is None:
        sim_plate = np.zeros((HEIGHT_PLATE, WIDTH_PLATE,3), np.uint8)
    else:
        sim_plate = background
    
    rad = 3

    for i, r in enumerate(ROW1):
        filled = -1 if row1[i] else 0
        cv2.circle(sim_plate, r, rad, (0, 0, 255), filled)
    for i, r in enumerate(ROW2):
        filled = -1 if row2[i] else 0
        cv2.circle(sim_plate, r, rad, (0, 0, 255), filled)
    for i, r in enumerate(ROW3):
        filled = -1 if row3[i] else 0
        cv2.circle(sim_plate, r, rad, (0, 0, 255), filled)
    for i, r in enumerate(ROW4):
        filled = -1 if row4[i] else 0
        cv2.circle(sim_plate, r, rad, (0, 0, 255), filled)

    return sim_plate

def order_points(pts):

	rect = np.zeros((4, 2), dtype = "float32")
	s = pts.sum(axis = 1)

	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	return rect

def four_point_transform(image, pts):

    global WIDTH_PLATE, HEIGHT_PLATE
    maxWidth = WIDTH_PLATE
    maxHeight = HEIGHT_PLATE
	# obtain a consistent order of the points and unpack them
	# individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    # widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    # maxWidth = max(int(widthA), int(widthB))

    # heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    # heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    # maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped

def aruco_classification(gray_plate, threshold=100, len_sample=3):

    global ROW1, ROW2, ROW3, ROW4

    row1 = []
    for r in ROW1:
        sample = gray_plate[r[1] - len_sample:r[1] + len_sample, r[0] - len_sample:r[0] + len_sample]
        sample = sample.flatten()
        row1.append(np.sum(sample)/(len(sample)) > threshold)

    row2 = []
    for r in ROW2:
        sample = gray_plate[r[1] - len_sample:r[1] + len_sample, r[0] - len_sample:r[0] + len_sample]
        sample = sample.flatten()
        row2.append(np.sum(sample)/(len(sample)) > threshold)

    row3 = []
    for r in ROW3:
        sample = gray_plate[r[1] - len_sample:r[1] + len_sample, r[0] - len_sample:r[0] + len_sample]
        sample = sample.flatten()
        row3.append(np.sum(sample)/(len(sample)) > threshold)

    row4 = []
    for r in ROW4:
        sample = gray_plate[r[1] - len_sample:r[1] + len_sample, r[0] - len_sample:r[0] + len_sample]
        sample = sample.flatten()
        row4.append(np.sum(sample)/(len(sample)) > threshold)
    
    # print(row1, row2, row2, row4)
    return row1, row2, row3, row4

def aruco_display(detector, image):

    corners, ids, rejected = detector.detectMarkers(image)
    ori_image = copy.deepcopy(image)

    list_corners = []

    if len(corners) > 0:
        # flatten the ArUco IDs list
        ids = ids.flatten()
        # loop over the detected ArUCo corners
        for (markerCorner, markerID) in zip(corners, ids):
            # extract the marker corners (which are always returned in
            # top-left, top-right, bottom-right, and bottom-left order)
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners
            # convert each of the (x, y)-coordinate pairs to integers
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))

            list_corners.append(topLeft)
            list_corners.append(topRight)
            list_corners.append(bottomLeft)
            list_corners.append(bottomRight)

            cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
            cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
            cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)
            # compute and draw the center (x, y)-coordinates of the ArUco
            # marker
            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
            # draw the ArUco marker ID on the image

            # cv2.putText(image, str(markerID),(topLeft[0], topLeft[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
            #     0.5, (0, 255, 0), 2)
            # print("[Inference] ArUco marker ID: {}".format(markerID))
            # show the output image

    if (len(list_corners) < 4*4):
        print("Not enough 4 aruco")
        return image, None

    rect = cv2.minAreaRect(np.array(list_corners))
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    # print(box)

    # draw_aruco_box(ori_image, box[0], box[3], box[1], box[2])
    tmp = four_point_transform(ori_image, box)
    gray = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("Haha",gray)

    return image, gray