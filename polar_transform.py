import numpy as np
import cv2
from cap_multithreading import CapMultiThreading
from numba import njit, prange
from tqdm import tqdm
import matplotlib.pyplot as plt


import cv2
import numpy as np

@njit(parallel=True)
def find_max(image, coords):
    for i in prange(image.shape[0]):
        coords[i] = np.argmax(image[i])

    return coords

@njit(parallel=True)
def cart_to_polar(image, cx, cy, polar_image):
    height, width = polar_image.shape
    for theta in prange(polar_image.shape[0]):
        angle = 2 * np.pi * theta/polar_image.shape[0]
        for r in prange(polar_image.shape[1]):
            
            x = int(round(cx + r * np.cos(angle)))
            y = int(round(cy + r * np.sin(angle)))

            if 0 <= x < width and 0 <= y < height:
                polar_image[theta, r] = image[y, x]
            else:
                polar_image[theta, r] = 0
        
    return polar_image

def transform(file_name, cent_func, cent_args, interpolation=None, scale=1):
    """A funcion to perform polar transform on the image around a trial centre and measure how well the rings
    form straight lines before adjusting the centre
    Inputs:
        file_name - the name of the video file to be used
        cent_func - the function to be used to find the initial centre guesses
        cent_args - arguments for cent_func
        interpolation - the interpolation algorithm to be used if resizing the image
        scale - the factor by which to resize the image """
    centres = cent_func(*cent_args)
    cap = CapMultiThreading(file_name)
    no_frames = cap.get_frame_count()
    centres_polar = []

    for i in tqdm(range(no_frames)):
        ret, frame = cap.get_frame()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)/255.0
        cx, cy = int(np.rint(centres[i][0])), int(np.rint(centres[i][1]))

        if interpolation == "bilinear":
            cX, cY = cx * scale, cy * scale
            bin_frame = cv2.resize(gray_frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            
        elif interpolation == "bicubic":
            cX, cY = cx * scale, cy * scale
            bin_frame = cv2.resize(gray_frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        else:
            bin_frame = gray_frame
            cX, cY = cx, cy
 
        max_radius = min(cX, cY)
        rmses = np.empty((5*scale, 5*scale))
        # loop over x, then y
        for j in range(5*scale):
            for k in range(5*scale):
                polar_image = cv2.linearPolar(bin_frame, (cX- 2*scale + j, cY- 2*scale + k), max_radius, cv2.WARP_FILL_OUTLIERS)
                coords = np.empty(polar_image.shape[0])
                coords = find_max(polar_image, coords)
                res = coords - np.mean(coords)
                rmses[j, k] = np.sqrt(np.mean(res**2))
            
        # find the lowest rmse in the matrix then perform fits in x and y
        min_coords = np.unravel_index(np.argmin(rmses), rmses.shape)
        y, x = min_coords[1], min_coords[0]
        x_points = np.arange(cX - 2*scale, cX + 3*scale)
        y_points = np.arange(cY - 2*scale, cY + 3*scale)
        
        xfit = np.polyfit(x_points, rmses[:,x], 2)
        yfit = np.polyfit(y_points, rmses[y], 2)

        x_poly = np.poly1d(xfit)
        y_poly = np.poly1d(yfit)

        x_deriv = x_poly.deriv()
        y_deriv = y_poly.deriv()

        x_sol = np.roots(x_deriv)
        y_sol = np.roots(y_deriv)

        centres_polar.append((x_sol/scale, y_sol/scale))    

    return np.array(centres_polar)

