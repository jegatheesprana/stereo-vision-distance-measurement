import cv2
import numpy as np


def plot_distance(org_image, lines):
    image = org_image.copy()
    for line in lines:
        x1 = int(line[0][0])
        y1 = int(line[0][1])
        x2 = int(line[1][0])
        y2 = int(line[1][1]*960/720)
        distance = np.around(line[2].numpy(), decimals=2)
        cv2.line(image, (x1, y1), (x2, y2), (50, 127, 205), 9)
        cv2.rectangle(image, ((x1+x2)//2, y2-70),
                      ((x1+x2)//2+150, y2-20), (50, 127, 205), -1)
        cv2.putText(image, str(distance) + " m", ((x1+x2)//2, y2-35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4)
    return image
