import cv2
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from cap_multithreading import CapMultiThreading
from numba import njit, prange
from tqdm import tqdm

cv2.setUseOptimized(True)

def fit_gradient(coord, data, background, scale, i, inner=False):
    """A function to fit a straight line to the max gradient detected
    Inputs:
        coord - the value at which the peak of the curve is detected
        data - the curve datapoints
        background - the background noise value to be used to calculate intercepts
        scale - the factor used for interpolation
        i - the current iteration (useful for debugging)
        inner - whether the gradient is inner i.e. whether the peak is corresponding to the left or right
    Outputs:
        intercept - the calculated intercept (divided by appropriate factor)"""
    
    # if inner is false it is the peak on the "right" so we would want an increasing grad (so positive)
    # also truncate the data to get the line on the right side of the peak
    if scale > 1:
        f = round(scale/2)
    else:
        f = 1
    
    if inner is False:
        truncated_data = data[coord - 4 * f: coord + 1]
        max_coord = np.argmax(np.diff(truncated_data)) + coord - 4 * f
        plot_coords = np.arange(max_coord - 3, max_coord + 2, 1)
        plot_data = data[max_coord - 3: max_coord + 2]
    else:
        truncated_data = data[coord: coord + 5 * f]
        min_coord = np.argmin(np.diff(truncated_data)) + coord
        plot_coords = np.arange(min_coord - 1, min_coord + 4, 1)
        plot_data = data[min_coord - 1: min_coord + 4]

    # generating the coords associated with the truncated data
    fit = np.polyfit(plot_coords, plot_data, 1)
    intercept = (background - fit[1])/fit[0]    

  

    return intercept/scale

def gradient_full(com_func, com_args, mode_args, fit, interpolation=None, scale=2):
    """Find centres using gradient method. Uses COM to create strips, then locate peaks in the intensity
    Use the peaks to find and fit a line to the steepest slopes and calculate the intercept points with
    background noise. Take the centre as the midway point between the two intercepts. Finds the z values using 
    a pregenerated fit of width and z from the lookup table.
    Inputs:
        cap - the video capture object
        no_frames - number of frames in the video
        com_func - COM func to use to estimate centres to take strips
        com_args - args to pass into COM func
        height - half the vertical height of the horizontal strip (will be used as the width of vertical strip)
        width - half the horizontal length of the strip (will be used as height of the vertical strip)
        interpolation - the interpolation algorithm to be used if resizing the image
        scale - the factor by which to resize the image 
    Outputs:
        centres_grad - an array of shape (no_frames, 2) containing (cx, cy) for each frame
        """
    centres_com = com_func(*com_args)
    coords_grad = []
    cap = CapMultiThreading(com_args[0])
    no_frames = cap.get_frame_count()
    height, width = mode_args

    for i in tqdm(range(no_frames)):
        ret, frame = cap.get_frame()
        if frame is None:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cx, cy = int(np.rint(centres_com[i][0])), int(np.rint(centres_com[i][1]))


        # finding initial peaks in the curve so they don't get affected by interpolation later
        h_strip_init = gray_frame[cy - height:cy + height + 1, cx - width:cx + width + 1]
        h1d_init = np.average(h_strip_init, axis=0)
        h1d_init -= np.median(h1d_init)

        v_strip_init = gray_frame[cy - width:cy + width + 1, cx - height:cx + height + 1]
        v1d_init = np.average(v_strip_init, axis=1)
        v1d_init -= np.median(v1d_init)


        x_points, _ = sp.signal.find_peaks(h1d_init, height=np.max(h1d_init)//2, distance=15)
        y_points, _ = sp.signal.find_peaks(v1d_init, height=np.max(v1d_init)//2, distance=15)
        
        if y_points.shape[0] != 2 or x_points.shape[0] !=2:
            print(f"x_points = {x_points}, y_points {y_points}")
            plt.figure()
            plt.plot(v1d_init)
            plt.show()

        if interpolation == "bilinear":
            bin_frame = cv2.resize(gray_frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            cX, cY = cx * scale, cy * scale
            w, h = width * scale, height * scale

        elif interpolation == "bicubic":
            bin_frame = cv2.resize(gray_frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            cX, cY = cx * scale, cy * scale
            w, h = width * scale, height * scale

        else:
            bin_frame = gray_frame
            cX, cY = cx, cy
            w, h = width, height
            # since we don't want to scale it up later if the image is not interpolated
            scale = 1

        background = np.median(bin_frame)

        # indexing the com centres and creating strips
        horiz_strip = bin_frame[cX - h:cY + h + 1, cX - w:cX + w + 1]
        vert_strip = bin_frame[cY - w:cY + w + 1, cX - h:cX + h + 1]
        horiz1d = np.average(horiz_strip, axis=0)
        vert1d = np.average(vert_strip, axis=1)
        

        xMin = fit_gradient(int(x_points[0] * scale), horiz1d, background, scale, i, inner=True)
        xMax = fit_gradient(int(x_points[1] * scale), horiz1d, background, scale, i)

        yMin = fit_gradient(int(y_points[0] * scale), vert1d, background, scale, i, inner=True)
        yMax = fit_gradient(int(y_points[1] * scale), vert1d, background, scale, i)

        gap_x = xMax - xMin

        if interpolation is None:
            # have to multiply by scale since default is scale = 2, and the intercepts get divided by it
            cx = (cX - w + (xMax - xMin) * scale/2 + xMin * scale)
            cy = (cY - w + (yMax - yMin) * scale/2 + yMin * scale)
            coords_grad.append((cx , cy))
        else:
            cx = (cX/scale - w/scale + (xMax - xMin)/2 + xMin)
            cy = (cY/scale - w/scale + (yMax - yMin)/2 + yMin)
            z = fit[0] * gap_x + fit[1]
            coords_grad.append((cx, cy, z))
            

        
        
    
    return np.array(coords_grad)