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
        inner - whether the gradient is inner i.e. whether the peak is corresponding to the left or right"""
    
    # if inner is false it is the peak on the "right" so we would want an increasing grad (so positive)
    # also truncate the data to get the line on the right side of the peak
    if scale > 1:
        f = round(scale/2)
    else:
        f = 1
    
    if inner == False:
        truncated_data = data[coord - 4 * f: coord + 1]
        max_coord = np.argmax(np.diff(truncated_data)) + coord - 4 * f
        plot_coords = np.arange(max_coord - 2, max_coord + 3, 1)
        plot_data = data[max_coord - 2: max_coord + 3]
    else:
        truncated_data = data[coord: coord + 5 * f]
        min_coord = np.argmin(np.diff(truncated_data)) + coord
        plot_coords = np.arange(min_coord - 2, min_coord + 3, 1)
        plot_data = data[min_coord - 2: min_coord + 3]

    # generating the coords associated with the truncated data
    fit = np.polyfit(plot_coords, plot_data, 1)
    intercept = (background - fit[1])/fit[0]

    return intercept/scale

def fit_quadratic(coord, data, scale):
    f = round(scale/3)
    trunc_data = data[coord * scale - 2 * f: coord * scale + 3 * f]
    plot_coords = np.arange(coord * scale - 2 * f, coord * scale + 3 * f)
    fit = np.polyfit(plot_coords, trunc_data, deg=2)
    poly = np.poly1d(fit)
    deriv = poly.deriv()
    peak = np.roots(deriv)

    return peak[0]


def generate_LUT(file_name, cent_function, cent_args, mode, mode_args, heights, teth=False, interpolation=None, scale=2):
    """A function to use the LUT video to generate the LUT to use for z tracking
    Inputs:
        file_name - name of the video file to be used
        cent_function - the function to use to estimate centre
        cent_args - args to pass into centre estimate function
        mode - the mode to generate the LUT
        heights - data of known height values from LUT video
        interpolation - the interpolation algorithm to be used if resizing the image
        scale - the factor by which to resize the image 
    Outputs:
        fit - fit parameters (grad method)
        profs - array of stored profiles (rad method)
        z - array of z values corresponding to stored profiles (rad method)
    """
    
    centres_com = cent_function(*cent_args)
    z = [] # to store the z values that get used for LUT generation

    cap = CapMultiThreading(file_name)
    no_frames = cap.get_frame_count()

    if mode == "grad":
        height, width, deg = mode_args
        widths = []
        for i in tqdm(range(no_frames)):
            ret, frame = cap.get_frame()
            if frame is None:
                break

            if i > len(heights):
                break

            # check every 10 frames for each new step, or special case for the first step which has 5 frames
            if (i%10 == 0 and i > 5 and abs(float(heights[i]) - float(heights[i-9])) > 0.04) or (i <= 5  and i%5==0):
                
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                cx, cy = int(np.rint(centres_com[i][0])), int(np.rint(centres_com[i][1]))

               

                h_strip_init = gray_frame[cy - height:cy + height + 1, cx - width:cx + width + 1]
                h1d_init = np.average(h_strip_init, axis=0)

                
                # get peak locations before interpolation, and scale them after
                x_points, _ = sp.signal.find_peaks(h1d_init, height=np.max(h1d_init) - 25)

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
                    scale = 1

                horiz_strip = bin_frame[cX - h:cY + h + 1, cX - w:cX + w + 1]
                horiz1d = np.average(horiz_strip, axis=0)
                horiz1d = sp.signal.savgol_filter(horiz1d, window_length=15, polyorder=2)

                background = np.median(bin_frame)

                
                # get the intercept points
                xMin = fit_gradient(int(x_points[0]) * scale, horiz1d, background, scale, i, inner=True)
                xMax = fit_gradient(int(x_points[1]) * scale, horiz1d, background, scale, i=i)

                gap = xMax - xMin
                widths.append(gap)
                
                z.append(heights[i])
                

        widths = np.array(widths)
        # tethered videos have beads get smaller as extension increases (moves away from camera)
        # for LUT video the beads get smaller as the objective is moved closer to the bead
        if teth is True:
            widths = widths[::-1]
        fit = np.polyfit(widths, np.array(z), deg)

        return fit

    elif mode == "rad":
        profs = []
        height, width = mode_args
        
        for i in tqdm(range(no_frames)):
            ret, frame = cap.get_frame()
            if frame is None:
                break
                
            # since the LUT video has more frames than data
            
            if i > len(heights):
                break
            
            if (i%10 == 0 and i > 5 and abs(float(heights[i]) - float(heights[i-9])) > 0.04) or (i <= 4  and i%4==0):
                
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                cx, cy = int(np.rint(centres_com[i][0])), int(np.rint(centres_com[i][1]))
                
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
                    scale = 1
                

                
                # taking radial profiles on right and left hand side of the centre    
                prof_x = np.average(bin_frame[cY - 1: cY + 2, cX: cX + 40 * scale], axis=0)
                px = np.average(bin_frame[cY - 1: cY + 2, cX - 39*scale: cX + 1 * scale], axis=0)[::-1]
            
                profs.append((prof_x + px)/2)
                
                z.append(heights[i])

        return np.array(profs), np.array(z) 


@njit(parallel=True)
def chi_sq(obs_prof, rad_profiles, chi):
    """A function to calculate the chi squared value using numba parallelisation
    Inputs:
        obs_prof - array of observed values, in this case the observed radial intensity profile
        rad_profiles - array containing all the radial intensity profiles from the lookup table
        chi - empty array to write the chi squared values into
    Outputs:
        chi - array of chi squared values"""
    for i in prange(rad_profiles.shape[0]):
        chi[i] = np.sum(((obs_prof - rad_profiles[i])**2)/rad_profiles[i])

    return chi
    


def z_position(file_name, cent_function, cent_args, mode, mode_args, interpolation=None, scale=2):
    """A function to calculate the z position of a bead in a video given certain predetermined fit parameters
    from a LUT
    Inputs:
        file_name - the name of the video to be used
        cent_function - the COM function to use to estimate centre
        cent_args - args to pass into COM func
        mode - the mode of z tracking to use
        deg - the order of polynomial fit used during LUT generation. (either 1 or 2)
        interpolation - the interpolation algorithm to be used if resizing the image
        scale - the factor by which to resize the image 
        """
    centres_com = cent_function(*cent_args)
    z_data = []
    gaps = []
    cap = CapMultiThreading(file_name)
    no_frames = cap.get_frame_count()


    # using the widths
    if mode == "grad":
        fit, height, width, deg = mode_args

        for i in tqdm(range(no_frames)):
            ret, frame = cap.get_frame()
            if frame is None:
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            background = np.median(gray_frame)

            cx, cy = int(np.rint(centres_com[i][0])), int(np.rint(centres_com[i][1]))

            h_strip_init = gray_frame[cy - height:cy + height + 1, cx - width:cx + width + 1]
            h1d_init = np.average(h_strip_init, axis=0)

            # get peak locations before interpolation, and scale them after
            x_points, _ = sp.signal.find_peaks(h1d_init, height=np.max(h1d_init) - 25)

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
                x_points = x_points/scale

            horiz_strip = bin_frame[cX - h:cY + h + 1, cX - w:cX + w + 1]
            horiz1d = np.average(horiz_strip, axis=0)
            horiz1d = sp.signal.savgol_filter(horiz1d, window_length=15, polyorder=2)

            background = np.median(bin_frame)

            # get the intercept points
            xMin = fit_gradient(int(x_points[0]) * scale, horiz1d, background, scale, i, inner=True)
            xMax = fit_gradient(int(x_points[1]) * scale, horiz1d, background, scale, i=i)

            gap = xMax - xMin

            gaps.append(gap)

            if deg == 1:
                z = fit[0] * gap + fit[1] 
            elif deg == 2:
                z = fit[0] * gap**2 + fit[1] * gap + fit[2]
        
            z_data.append(z)


        return np.array(z_data)

    if mode == "rad":
        profs, z_lut = mode_args
        for i in tqdm(range(no_frames)):
            ret, frame = cap.get_frame()
            if frame is None:
                break
            
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            background = np.median(gray_frame)

            cx, cy = int(np.rint(centres_com[i][0])), int(np.rint(centres_com[i][1]))

            if interpolation == "bilinear":
                bin_frame = cv2.resize(gray_frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                cX, cY = cx * scale, cy * scale

            elif interpolation == "bicubic":
                bin_frame = cv2.resize(gray_frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                cX, cY = cx * scale, cy * scale

            else:
                bin_frame = gray_frame
                cX, cY = cx, cy
                scale = 1
            

            # getting the radial profiles
            prof_x = np.average(bin_frame[cY - 1: cY + 2, cX: cX + 40 * scale], axis=0)
            px = np.average(bin_frame[cY - 1: cY + 2, cX - 39*scale: cX + 1 * scale], axis=0)[::-1]
            #prof_x = np.average(bin_frame[cY - 1: cY + 2, cX - 40: cX + 41], axis=0)

            # average over both of the x directions (should be kept same as what was used in the LUT gen)
            p = (prof_x + px)/2
            #p = prof_x
            # generating chi values compared to all LUT profiles, and finding the minimum
            chi = np.empty(profs.shape[0])
            chi = chi_sq(p, profs, chi)
            min_chi = np.argmin(chi)
            
            # get 5 points around the minimum value of chi to fit quadratic for sub step accuracy
            start_index = min(max(min_chi - 5//2, 0), z_lut.shape[0] - 5)

            trunc_chi = chi[start_index: start_index + 5]
            trunc_z = z_lut[start_index: start_index + 5]
            fit = np.polyfit(trunc_z, trunc_chi, deg=2)

            # solve the fitted quadratic to find the z value
            chi_poly = np.poly1d(fit)
            chi_deriv = chi_poly.deriv()
            chi_sol = np.roots(chi_deriv)

            z_data.append(chi_sol)

            if i == 360 or i % 500==0:
                fig, ax = plt.subplots(1, 2)
                ax[0].plot(z_lut, chi)
                ax[0].set_xlabel("Z (μm)")
                ax[0].set_ylabel("$\chi^2$")
                ax[0].set_title("$\chi^2$ values for one frame")
                
                ax[1].plot(trunc_z, trunc_chi, "-*", label="Calculated $\chi^2$")
                ax[1].plot(trunc_z, np.polyval(fit, trunc_z), label="fitted quadratic")
                ax[1].legend(loc='best')
                ax[1].set_xlabel("Z (μm)")
                ax[1].set_ylabel("$\chi^2$")
                ax[1].set_title("Points around minimum $\chi^2$ value")
                plt.show()

        
            
        return np.array(z_data)






