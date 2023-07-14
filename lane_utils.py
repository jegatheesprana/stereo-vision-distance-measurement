from ultrafastLaneDetector import UltrafastLaneDetector, ModelType
import numpy as np
from depth import calculate_depth


class LaneDetector:
    def __init__(self, lane_model_path="/content/drive/Shareddrives/FYP/Source_codes/lane_detection/tusimple_18.pth"):
        lane_model_type = ModelType.TUSIMPLE
        use_gpu = True

        # Initialize lane detection model
        self.lane_detector = UltrafastLaneDetector(
            lane_model_path, lane_model_type, use_gpu)

    def detect_lanes(self, image):
        input_tensor = self.lane_detector.prepare_input(image)

        # Perform inference on the image
        output = self.lane_detector.inference(input_tensor)

        # Process output data
        lanes_points, lanes_detected = self.lane_detector.process_output(
            output, self.lane_detector.cfg)

        return lanes_points, lanes_detected

    def draw_lanes(self, input_img, lanes_points, lanes_detected):
        # Draw depth image
        visualization_img = self.lane_detector.draw_lanes(
            input_img, lanes_points, lanes_detected, self.lane_detector.cfg, True)

        return visualization_img


def find_depth_for_lanes(lanes_points, lanes_detected, disp):
    lanes_with_depth = []
    for lane in lanes_points:
        lane_info = []
        for lane_point in lane:
            depth = calculate_depth(
                int(lane_point[0]), int(lane_point[1]*960/720), disp)
            lane_info.append([lane_point[0], lane_point[1], depth])
        lanes_with_depth.append(lane_info)
    return lanes_with_depth


def calculate_distance(x1, x2, d1, d2):
    focal_length = 0.08
    dx = np.abs(x2 - x1)*4860*10**(-6)
    theta = np.arctan(dx / focal_length)
    D = np.sqrt((d1**2) + (d2**2) - (2 * d1 * d2 * np.cos(theta)))
    return D


def find_lane_point_for_objects(objects, lanes_with_depth):
    lines = []
    for box, depth in objects:
        x1, y1, x2, y2 = box.xyxy[0]

        bottom_center_x = (x1+x2)//2
        bottom_center_y = int(y2.cpu())

        horizontal_points = []
        class_name = box.class_detect

        if (class_name not in ['bicycle', 'car', 'dog', 'motorbike', 'person', 'three wheeler', 'vehicle']):
            for lane in lanes_with_depth:
                if (len(lane) > 0):
                    lane_depths = np.array(lane)[:, 1]
                    horizontal_point_id = (
                        np.abs(lane_depths - int(bottom_center_y*720/960))).argmin()
                    # lane_depths = np.array(lane)[:, 2]
                    # horizontal_point_id = (np.abs(lane_depths - depth)).argmin()
                    if (horizontal_point_id >= 0):
                        horizontal_point = lane[horizontal_point_id]
                        distance = calculate_distance(bottom_center_x.cpu(
                        ), horizontal_point[0], depth, horizontal_point[2])
                        horizontal_points.append(
                            [horizontal_point[0], horizontal_point[1], distance])
                        # horizontal_points.append([horizontal_point, distance])
        if len(horizontal_points) > 0:
            distance_arr = np.array(horizontal_points)[:, 0]
            nearest_point_id = (
                np.abs(distance_arr - int(bottom_center_x))).argmin()
            nearest_point = horizontal_points[nearest_point_id]

            lines.append([[bottom_center_x, bottom_center_y], [
                         nearest_point[0], nearest_point[1]], nearest_point[2]])
            # lines.append([[bottom_center_x, bottom_center_y], [nearest_point[0][0], nearest_point[0][1]], nearest_point[1]])

    return lines
