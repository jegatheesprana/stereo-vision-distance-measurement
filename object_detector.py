from ultralytics import YOLO
from ultralytics.yolo.utils.plotting import Annotator
import math
import cv2
import numpy as np

# Create a list of colors for each class where each color is a tuple of 3 integer values
rng = np.random.default_rng(3)
colors = rng.uniform(0, 255, size=(80, 3))


class ObjectDetector:
    def __init__(self, model_path="/content/drive/Shareddrives/FYP/Source_codes/Object_detection/FYP-Final-V1-2/runs/detect/train7/weights/best.pt"):
        self.model = YOLO(model_path)

    def predict_objects(self, image):
        results = self.model.predict(image, agnostic_nms=True)

        for info in results:
            parameters = info.boxes
            for box in parameters:
                class_detect = self.model.names[int(box.cls[0])]
                box.class_detect = class_detect
        return results

    @staticmethod
    def draw_objects(old_frame, results):
        frame = old_frame.copy()
        annotator = Annotator(frame, 4, 4)
        for box, distance in results:
            x1, y1, x2, y2 = box.xyxy[0]
            confidence = box.conf[0]
            conf = math.ceil(confidence * 100)
            class_detect = box.class_detect
            if conf >= 0:
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                color = [int(c) for c in colors[int(box.cls[0])]]
                # cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
                # cv2.putText(frame, class_detect, (x1-10,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 4)
                # get box coordinates in (top, left, bottom, right) format
                b = box.xyxy[0]
                annotator.box_label(b, class_detect, color=color)
                # cv2.rectangle(frame, (x1, y2+5), (x1+130, y2+40), color, -1)
                # cv2.putText(frame, str(distance)+ " m", (x1+5,y2+35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 4)
                cv2.circle(frame, (int((x1+x2)/2), int((y1+y2)/2)),
                           radius=10, color=(255, 255, 255), thickness=-1)
                cv2.circle(frame, (int((x1+x2)/2), int((y1+y2)/2)),
                           radius=5, color=color, thickness=-1)
        return frame
