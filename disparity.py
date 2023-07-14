import cv2
import numpy as np

# Camera parameters to undistort and rectify images
cv_file = cv2.FileStorage()
cv_file.open(
    '/content/drive/Shareddrives/FYP/Source_codes/stereo_vision/stereoMap.xml', cv2.FileStorage_READ)

stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()
R = cv_file.getNode('rot').mat()
T = cv_file.getNode('trans').mat()

# *******************************************
# ***** Parameters for the StereoVision *****
# *******************************************

# Create StereoSGBM and prepare all parameters
window_size = 3
min_disp = 2
num_disp = 130-min_disp
stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
                               numDisparities=num_disp,
                               blockSize=window_size,
                               uniquenessRatio=10,
                               speckleWindowSize=100,
                               speckleRange=32,
                               disp12MaxDiff=5,
                               P1=8*3*window_size**2,
                               P2=32*3*window_size**2)

# Used for the filtered image
# Create another stereo for right this time
stereoR = cv2.ximgproc.createRightMatcher(stereo)

# WLS FILTER Parameters
lmbda = 80000
sigma = 1.8
visual_multiplier = 1.0

wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)

kernel = np.ones((3, 3), np.uint8)


def find_disparity(frame_left, frame_right):
    left_rectified = cv2.remap(
        frame_left, stereoMapL_x, stereoMapL_y, interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT)
    right_rectified = cv2.remap(
        frame_right, stereoMapR_x, stereoMapR_y, interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT)

    # Convert from color(BGR) to gray
    grayR = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)
    grayL = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)

    # Compute the 2 images for the Depth_image
    disp = stereo.compute(grayL, grayR)  # .astype(np.float32)/ 16
    dispL = disp
    dispR = stereoR.compute(grayR, grayL)
    dispL = np.int16(dispL)
    dispR = np.int16(dispR)

    # Using the WLS filter
    filteredImg = wls_filter.filter(dispL, grayL, None, dispR)
    filteredImg = cv2.normalize(
        src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
    filteredImg = np.uint8(filteredImg)
    # cv2.imshow('Disparity Map', filteredImg)
    # Calculation allowing us to have 0 for the most distant object able to detect
    disp = ((disp.astype(np.float32) / 16)-min_disp)/num_disp

# Resize the image for faster executions
# dispR= cv2.resize(disp,None,fx=0.7, fy=0.7, interpolation = cv2.INTER_AREA)

    # Filtering the Results with a closing filter
    # Apply an morphological filter for closing little "black" holes in the picture(Remove noise)
    closing = cv2.morphologyEx(disp, cv2.MORPH_CLOSE, kernel)

    # Colors map
    dispc = (closing-closing.min())*255
    # Convert the type of the matrix from float32 to uint8, this way you can show the results with the function cv2.imshow()
    dispC = dispc.astype(np.uint8)
    # Change the Color of the Picture into an Ocean Color_Map
    disp_Color = cv2.applyColorMap(dispC, cv2.COLORMAP_OCEAN)
    filt_Color = cv2.applyColorMap(filteredImg, cv2.COLORMAP_OCEAN)

    # return filt_Color, disp
    return left_rectified, filt_Color, disp
