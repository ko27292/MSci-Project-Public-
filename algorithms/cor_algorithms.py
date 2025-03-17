import cv2
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from cap_multithreading import CapMultiThreading
from numba import njit, prange
from tqdm import tqdm
from matplotlib.patches import Rectangle

cv2.setUseOptimized(True)

def cor_mirror_strip(com_func, com_args, height, width, peak_width=2, deg=2, interpolation="bilinear", scale=1):
    """A function to perform cross-correlation. First uses COM to determine rough centres, before taking 
    horizontal and vertical strips and cross correlating these strips with their mirror image to find x and y.
    Inputs:
        com_func - the com function to be used
        com_args - args to pass into COM func
        height - half the vertical height of the horizontal strip (will be used as the width of vertical strip)
        width - half the horizontal length of the strip (will be used as height of the vertical strip)
        peak_width - the width of peak to fit to in correlation curve. standard is to fit to 5 points (e.g. width of 2)
        deg - the order of polynomial to fit to in correlation curve. standard is a quadratic (deg = 2)
        interpolation - the interpolation algorithm to be used if resizing the image
        scale - the factor by which to resize the image 
    Outputs:
        centres_cor - array of the x, y pairs for the centre as determined by COR algorithm
        """
    
    centres_com = com_func(*com_args)
    centres_cor = []

    cap = CapMultiThreading(com_args[0])
    no_frames = cap.get_frame_count()

    for i in tqdm(range(no_frames)):
        ret, frame = cap.get_frame()
        if frame is None:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # round the com coords to ints so they can be used for array slicing
        cx, cy = int(np.rint(centres_com[i][0])), int(np.rint(centres_com[i][1]))

        # handling interpolation and scaling necessary variables
        if interpolation == "bilinear":
            pw = peak_width
            w, h = width * scale, height * scale
            cX, cY = cx * scale, cy * scale
            bin_frame = cv2.resize(gray_frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            
        elif interpolation == "bicubic":
            pw = peak_width
            w, h = width * scale, height * scale
            cX, cY = cx * scale, cy * scale
            bin_frame = cv2.resize(gray_frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        else:
            bin_frame = gray_frame
            pw, w, h = peak_width, width, height
            cX, cY = cx, cy


        # getting strips
        horiz_strip = bin_frame[cY - h:cY + h + 1, cX - w:cX + w + 1]
        vert_strip = bin_frame[cY - w:cY + w + 1, cX - h:cX + h + 1]
        horiz1d = np.average(horiz_strip, axis=0)
        vert1d = np.average(vert_strip, axis=1)
        
        # correlation
        corrx = sp.signal.correlate(horiz1d, horiz1d[::-1])
        corry = sp.signal.correlate(vert1d, vert1d[::-1])
        
        # finding the peak of each correlation 
        max_point_x, _ = sp.signal.find_peaks(corrx, height=np.max(corrx))
        max_point_y, _ = sp.signal.find_peaks(corry, height=np.max(corry))

        x = np.arange(max_point_x - pw, max_point_x + pw + 1, 1)
        y = np.arange(max_point_y - pw, max_point_y + pw + 1, 1)

        # this needs to be -x, +x+1 so it is still symmetric after indexing
        corrx_trunc = corrx[max_point_x[0] - pw: max_point_x[0] + pw + 1]
        corry_trunc = corry[max_point_y[0] - pw: max_point_y[0] + pw + 1]

        
        # fitting
        xfit = np.polyfit(x, corrx_trunc, deg)
        yfit = np.polyfit(y, corry_trunc, deg)

        x_poly = np.poly1d(xfit)
        y_poly = np.poly1d(yfit)

        x_deriv = x_poly.deriv()
        y_deriv = y_poly.deriv()

        x_sol = np.roots(x_deriv)
        y_sol = np.roots(y_deriv)

        # (corr.shape[0]/ - 1)/2 gives the point of symmetry
        x_rel = (x_sol - (corrx.shape[0] - 1)/2)/scale
        y_rel = (y_sol - (corry.shape[0] - 1)/2)/scale
    

        centres_cor.append((cx + x_rel, cy + y_rel))

        
    return np.array(centres_cor) 



    


