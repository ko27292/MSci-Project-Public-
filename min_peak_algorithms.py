import cv2
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from cap_multithreading import CapMultiThreading
from numba import njit, prange
from tqdm import tqdm

cv2.setUseOptimized(True)


def min_peak(com_func, com_args, height, width, peak_width=2, interpolation=None, scale=1):
    """A function to find the centre by fitting a quadratic to the small peak in intensity at the centre of the bead
    Inputs:
        com_func - the COM function to be used
        com_args - args to pass into COM func
        height - half the vertical height of the horizontal strip (will be used as the width of vertical strip)
        width - half the horizontal length of the strip (will be used as height of the vertical strip)
        peak_width - width of peak to be used for fitting
        interpolation - the interpolation algorithm to be used if resizing the image
        scale - the factor by which to resize the image
    Outputs:
        centres_mp - array of the x, y pairs for the centre as determined by minimum peak algorithm
        """
    centres_com = com_func(*com_args)
    centres_mp = []
    pw = peak_width
    cap = CapMultiThreading(com_args[0])
    no_frames = cap.get_frame_count()

    for i in tqdm(range(no_frames)):
        ret, frame = cap.get_frame()
        if frame is None:
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # indexing the com centres and creating strips
        cx, cy = int(np.rint(centres_com[i][0])), int(np.rint(centres_com[i][1]))

        h_strip_init = gray_frame[cy - height:cy + height + 1, cx - width:cx + width + 1]
        h1d_init = np.average(h_strip_init, axis=0)

        v_strip_init = gray_frame[cy - width:cy + width + 1, cx - height:cx + height + 1]
        v1d_init = np.average(v_strip_init, axis=1)

        x_point, _ = sp.signal.find_peaks(h1d_init, height=(np.min(h1d_init), np.min(h1d_init) + 35))
        y_point, _ = sp.signal.find_peaks(v1d_init, height=(np.min(v1d_init), np.min(v1d_init) + 35))


        # interpolation algorithm to rescale the image for better resolution
        if interpolation == "bilinear":
            w, h = width * scale, height * scale
            cX, cY = cx * scale, cy * scale
            bin_frame = cv2.resize(gray_frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            x_point, y_point = x_point * scale, y_point * scale
            
        elif interpolation == "bicubic":
            w, h = width * scale, height * scale
            cX, cY = cx * scale, cy * scale
            bin_frame = cv2.resize(gray_frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            x_point, y_point = x_point * scale, y_point * scale

        else:
            bin_frame = gray_frame
            pw, w, h = peak_width, width, height
            cX, cY = cx, cy


        horiz_strip = bin_frame[cY - h:cY + h + 1, cX - w:cX + w + 1]
        vert_strip = bin_frame[cY - w:cY + w + 1, cX - h:cX + h + 1]
        horiz1d = np.average(horiz_strip, axis=0)
        vert1d = np.average(vert_strip, axis=1)
        
        
        # getting the data around the central peak
        x_peak = horiz1d[x_point[0] - pw: x_point[0] + pw + 1]
        y_peak = vert1d[y_point[0] - pw: y_point[0] + pw + 1]

        # coords of the points around the peak
        x_coords = np.arange(x_point[0] - pw , x_point[0] + pw + 1)
        y_coords = np.arange(y_point[0] - pw , y_point[0] + pw + 1)
        
        #creating more points and interpolating
        x_interp = np.linspace(x_point[0] - pw, x_point[0] + pw, 5)
        y_interp = np.linspace(y_point[0] - pw ,y_point[0] + pw, 500)

        xp = np.interp(x_interp, x_coords, x_peak)
        yp = np.interp(y_interp, y_coords, y_peak)
        
        # fitting then finding stationary point of the quadratic 
        xfit = np.polyfit(x_interp, xp, deg=2)
        yfit = np.polyfit(y_interp, yp, deg=2)

        x_poly = np.poly1d(xfit)
        y_poly = np.poly1d(yfit)

        x_deriv = x_poly.deriv()
        y_deriv = y_poly.deriv()

        x_sol = np.roots(x_deriv)
        y_sol = np.roots(y_deriv)

        
        centres_mp.append(((cX - w + x_sol)/scale, (cY - w + y_sol)/scale))

        
    return np.array(centres_mp)