import cv2
import numpy as np
import scipy as sp
from cap_multithreading import CapMultiThreading
from numba import njit, prange
from tqdm import tqdm
import matplotlib.pyplot as plt

def blob(file_name, p, interpolation=None, scale=1):
    """A function to estimate the centre using openCV blob detector
    Inputs:
        file_name - the name of the video file to be used
        p - the parameters to be used for the blob detector setup
        interpolation - the interpolation algorithm to be used if resizing the image
        scale - the factor by which to resize the image
    Outputs:
        centres_blob -  array of the x, y pairs for the centre as determined by blob algorithm
    """
    
    cap = CapMultiThreading(file_name)
    no_frames = cap.get_frame_count()
    centres_blob = []

    # setting up blob detector params
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = p[0]
    params.maxThreshold = p[1]
    params.filterByCircularity = True
    params.minCircularity = p[2]

    detector = cv2.SimpleBlobDetector.create(params)

    # loop over frames of the video
    for i in tqdm(range(no_frames)):
        ret, frame = cap.get_frame()
        if frame is None:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # interpolation to scale up the image for better resolution
        if interpolation == "bilinear":
            bin_frame = cv2.resize(gray_frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            
        elif interpolation == "bicubic":
            bin_frame = cv2.resize(gray_frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        else:
            bin_frame = gray_frame

    
        keypoints = detector.detect(bin_frame)
        cx, cy = keypoints[0].pt
        

        centres_blob.append((cx/scale, cy/scale))  


    return np.array(centres_blob)
            

def canny_detect(file_name, args, interpolation=None, scale=1):
    """Algorithm to determine the centre by first using a canny edge detector, then running a blob detector on the edges
    Inputs:
        file_name - name of the video file to be used
        args - args to pass into canny edge and simple blob detector
    Outputs:
        centres_canny - array of the x, y pairs for the centre as determined by canny edge algorithm
    """
    cap = CapMultiThreading(file_name)
    no_frames = cap.get_frame_count()
    centres_canny = []

    for i in tqdm(range(no_frames)):
        ret, frame = cap.get_frame()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # interpolation algorithm to rescale the image for better resolution
        if interpolation == "bilinear":
            bin_frame = cv2.resize(gray_frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            
        elif interpolation == "bicubic":
            bin_frame = cv2.resize(gray_frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        else:
            bin_frame = gray_frame

        edges = cv2.Canny(bin_frame, args[0], args[1])
       
        # setting up blob detector parameters
        params = cv2.SimpleBlobDetector_Params()
        params.minThreshold = args[2]
        params.maxThreshold = args[3]
        params.filterByCircularity = True
        params.minCircularity = args[4]

        detector = cv2.SimpleBlobDetector.create(params)

        keypoints = detector.detect(edges)

        
        if len(keypoints) != 0:
            cx, cy = keypoints[0].pt

        centres_canny.append((cx/scale, cy/scale))
        
    return np.array(centres_canny)