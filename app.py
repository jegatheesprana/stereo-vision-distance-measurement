import cv2

from disparity import find_disparity
from depth import find_depths
from lane_utils import LaneDetector, find_depth_for_lanes, find_lane_point_for_objects
from object_detector import ObjectDetector
from distance import plot_distance


class Process:
    def __init__(self, calibration_path, yolo_model_path, lane_detection_model_path):
        self.calibration_path = calibration_path
        self.yolo_model_path = yolo_model_path
        self.lane_detection_model_path = lane_detection_model_path

        self.objectDetector = ObjectDetector(yolo_model_path)
        self.laneDetector = LaneDetector(lane_detection_model_path)

    def process_image(self, img):
        height, width, channels = img.shape

        frame_left = img[0:height, 0:int(width/2)]
        frame_right = img[0:height, int(width/2):width+1]

        # Hit "q" to close the window
        # if cv2.waitKey(1) & 0xFF == ord('a'):

        rectified_frame, filt_Color, disp = find_disparity(
            frame_left, frame_right)

        objects = self.objectDetector.predict_objects(rectified_frame)

        objects_with_dept = find_depths(objects, disp)

        # Lane detection

        lanes_points, lanes_detected = self.laneDetector.detect_lanes(
            rectified_frame)
        lanes_with_depth = find_depth_for_lanes(
            lanes_points, lanes_detected, disp)

        # Nearest Point calculation

        lines = find_lane_point_for_objects(
            objects_with_dept, lanes_with_depth)

        with_lane = self.laneDetector.draw_lanes(
            rectified_frame, lanes_points, lanes_detected)

        with_lane = cv2.resize(with_lane, (1280, 960))

        with_objects = self.objectDetector.draw_objects(
            with_lane, objects_with_dept)

        with_distance = plot_distance(with_objects, lines)

        return with_distance
