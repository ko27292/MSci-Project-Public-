import cv2
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

from cap_multithreading import CapMultiThreading
from numba import njit, prange
from tqdm import tqdm

cv2.setUseOptimized(True)

def com_single(file_name, threshold=10, interpolation=None, scale=1):
    """COM algorithm for use on a single bead video (no cropping to an ROI, all pixels in the video are used)
    Inputs:
        file_name - the name of the video file to be used
        threshold - value below which a pixel will be set to zero
        interpolation - the interpolation algorithm to be used if resizing the image
        scale - the factor by which to resize the image
    Outputs:
        centres - array of the x, y pairs for the centre as determined by COM algorithm"""
    centres = []
    cap = CapMultiThreading(file_name)
    no_frames = cap.get_frame_count()

    for i in tqdm(range(no_frames)):
        ret, frame = cap.get_frame()
        if frame is None:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
        # interpolation to scale up the image for better resolution
        if interpolation == "bilinear":
            bin_frame = cv2.resize(gray_frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            median = np.median(bin_frame)
            bin_frame = np.absolute(bin_frame - median)
    
        elif interpolation == "bicubic":
            bin_frame = cv2.resize(gray_frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            median = np.median(bin_frame)
            bin_frame = np.absolute(bin_frame - median)
            
        else:
            bin_frame = gray_frame
            median = np.median(bin_frame)
            bin_frame = np.absolute(bin_frame - median)
        
        # apply threshold and normalise
        bin_frame[bin_frame < threshold] = 0
        bin_frame = bin_frame/np.max(bin_frame)

        # setup and perform COM calculation
        x = np.arange(bin_frame.shape[1])
        y = np.arange(bin_frame.shape[0])
        xx, yy = np.meshgrid(x, y)
        sum_intensities = np.sum(bin_frame)
        cx = np.sum(xx * bin_frame)/sum_intensities
        cy = np.sum(yy * bin_frame)/sum_intensities

        centres.append((cx/scale, cy/scale))    

    return np.array(centres)