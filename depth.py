import numpy as np


def calculate_depth(x, y, disp):
    average = 0
    for u in range(-1, 2):
        for v in range(-1, 2):
            average += disp[y+u, x+v]
    average = average/9
    Depth = -593.97*average**(3) + 1506.8 * \
        average**(2) - 1373.1*average + 522.06
    Depth = np.around(Depth*0.01*0.35, decimals=2)
    return Depth


def find_depths(results, disp):
    new_results = []
    for info in results:
        parameters = info.boxes
        for box in parameters:
            x1, y1, x2, y2 = box.xyxy[0]
            x_avg = int((x1+x2)/2)
            y_avg = int((y1+y2)/2)

            depth = calculate_depth(x_avg, y_avg, disp)
            new_results.append([box, depth])
    return new_results
