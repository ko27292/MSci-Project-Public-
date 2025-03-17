import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import timeit
import pyximport
import scipy as sp
import sys
import gspread
import tensorflow as tf
import timeit

pyximport.install()

# import algorithms 
from cap_multithreading import CapMultiThreading
from algorithms.com_algorithms import com_single
from algorithms.cor_algorithms import cor_mirror_strip
from algorithms.gradient_algorithms import gradient_method
from algorithms.z_algorithms import generate_LUT, z_position
from polar_transform import transform
from algorithms.min_peak_algorithms import min_peak
from algorithms.blob_detect import blob, canny_detect
from algorithms.cnn import gen_data, dense, reload_dense, train_z
from algorithms.testing import gen_canny, train_dense, train_cnn, run_dense, run_cnn, train_z_cnn, cnn_z
from algorithms.grad_full_track import gradient_full

from numba import njit, jit, prange
from tqdm import tqdm
from gspread import Cell
from tensorflow import keras
from keras import Input


cv2.setUseOptimized(True)

def import_data():
    # loading data from spreadsheets
    gc = gspread.service_account(filename="msci-448512-32305bfe69de.json")
    sh = gc.open_by_url("https://docs.google.com/spreadsheets/d/11mkewTAp2WTARSqxbTK8l-oXrSUbi2sdm1pEw5THgOc/edit?gid=0#gid=0")
    lut = sh.sheet1
    lut_data = lut.get_all_records()
    lut_heights = [d["Piezo z position (microns)"] for d in lut_data][:706] # 706 as the frames after this are decreasing z (non-uniformly)

    lut_teth = sh.worksheet("Tethered")
    lut_teth_data = lut_teth.get_all_records()
    teth_heights = [d["Piezo z position"] for d in lut_teth_data][:706]

    fix_train = sh.worksheet("Fixed train")
    fix_data = fix_train.get_all_records()

    teth_train = sh.worksheet("Tethered train")
    teth_data = teth_train.get_all_records()

    temp = sh.worksheet("Tethered actual")
    temp_data = temp.get_all_records()
    temph = [d["z(um)"] for d in temp_data]

    centred_train = sh.worksheet("128 Fixed Train")
    cent_train = centred_train.get_all_records()
    
    return lut_heights, teth_heights, fix_data, teth_data, temph, fix_train, cent_train

def test_algorithm(algorithm, args, diff, video_name, show=True, ret=False):
    """Function used for testing x-y algorithms.
    Inputs:
        algorithm - the algorithm to be tested
        args - inputs for algorithm
        diff - the value expected for the shifts, in pixels
        video_name - name of the video used (for graph title)
        show - argument to display the graphs of position
        ret - whether to return arrays of the position
    """
    centres = algorithm(*args)
    start_index = 0
    prev_position = 0

    positions = []
    std = 1
    if show is True:
        for i in tqdm(range(centres.shape[0])):
            if i > 2:
                if abs(centres[i-1][0] - centres[i][0]) > diff or abs(centres[i-2][0] - centres[i][0]) > diff:
                
                    if abs(i - start_index) < 3:
                        continue
                    else:
                        if start_index == 0:
                            prev_position = np.average(centres[0:i][:,0])
                            start_index = i
                            std = np.std(centres[0:i][:,0])
                        else:
                            new_position = np.average(centres[start_index:i][:,0])
                            start_index = i
                            positions.append(abs(prev_position - new_position))
                            prev_position = new_position
            if i == centres.shape[0] - 1:
                new_position = np.average(centres[start_index:i][:,0])
                positions.append(abs(prev_position - new_position))

        print(np.array(positions), np.array(positions) * 142, np.average(positions) * 142)
        if len(positions) > 0:
            print(f"σ={std*142}nm, SNR={positions[0]/std}, σy = {np.std(centres[:,1])*142}nm")

        fig, ax = plt.subplots(1, 2)
        
        ax[0].plot(centres[:,0], linewidth=0.5)
        ax[0].set_title(f"X position vs time for {algorithm.__name__} ({video_name})")
        ax[0].set_xlabel("Frames")
        ax[0].set_ylabel("X position (px)")
    
        ax[1].plot(centres[:,1], linewidth=0.5)
        ax[1].set_title(f"Y position vs time for {algorithm.__name__} ({video_name})")
        ax[1].set_xlabel("Frames")
        ax[1].set_ylabel("Y position (px)")
        plt.show()

    if ret is True:
        return centres[:,0], centres[:,1]

def test_z(lut_args, args, diff, video_name, show=True, ret=False):
    """Function used for testing z algorithms.
    Inputs:
        lut_args - arguments to be used for the LUT generation
        args - inputs the z tracking algorithm
        diff - the value expected for the shifts, in μm
        video_name - name of the video used (for graph title)
        show - argument to display the graphs of position
        ret - whether to return arrays of the position
    """
    fit, *lut_res = generate_LUT(*lut_args)
    args[3].insert(0, fit) # inserting fit into the mode args list
    z_pos = z_position(*args)
    if show is True:
        positions = []
        index = []
        start_index = 0
        for i in range(z_pos.shape[0]):
            if i > 2:
                if abs(z_pos[i-1] - z_pos[i]) > diff or abs(z_pos[i-2] - z_pos[i]) > diff:
                
                    if abs(i - start_index) < 3:
                        continue
                    else:
                        if start_index == 0:
                            prev_position = np.average(z_pos[0:i])
                            start_index = i
                
                        else:
                            new_position = np.average(z_pos[start_index:i])
                            start_index = i
                            positions.append(abs(prev_position - new_position))
                            prev_position = new_position
            if i == z_pos.shape[0] - 1:
                new_position = np.average(z_pos[start_index:i])
                positions.append(abs(prev_position - new_position))

        print(np.array(positions))
    

        plt.figure()
        plt.plot(z_pos, linewidth=0.5, label="Calculated height (μm)")
        #plt.plot(teth_heights, label="Actual height (μm)")
        plt.legend(loc="best")
        plt.xlabel("frame count")
        plt.ylabel("z position")
        plt.title(f"Tracked z position vs frame count for {video_name}")
        plt.show()

    if ret is True:
        return np.array(z_pos)

def gen_train(algorithm, args, offset, sheet):
    """Function to generate and upload position data for training of NNs
    Inputs:
        algorithm - the algorithm to use to track position
        args - inputs for algorithm
        offset - which row to start adding data from
        sheet - the sheet to add data into
        """
    centres = algorithm(*args)
    
    
    cells = []
    for i in range(centres.shape[0]):
        cells.append(Cell(row=i+2+offset, col=1, value=centres[i][0])) # x value
        cells.append(Cell(row=i+2+offset, col=2, value=centres[i][1])) # y value
    
    sheet.update_cells(cells)
    train_data = sheet.get_all_records()
    x_data = [d["x"] for d in train_data]
    plt.figure()
    plt.plot(x_data, linewidth=0.5)
    plt.show()


#lut_heights, teth_heights, fix_data, teth_data, temph, fix_train, cent_train = import_data()


# arguments for 200nm shift video
com_args = ["videos/200nm shift every 5s cropped.avi"]
# 165, 225, 0.8 no interp, 1
blob_args = ["videos/200nm shift every 5s cropped.avi", [165, 225, 0.7]]
cor_args = [com_single, com_args, 8, 40, 2, 2]
mp_args = [com_single, com_args, 8, 50, 2]
grad_args = [com_single, com_args, 8, 50, "bilinear", 3]


#test_algorithm(com_single, com_args, 0.7, "200nm shift")

#200, 230, 200, 225, 0.8 no interp, 1
#test_algorithm(canny_detect, ["videos/200nm shift every 5s cropped.avi", (200, 230, 200, 225, 0.8)], 0.7, "200nm shift")
#test_algorithm(blob, blob_args, 0.7, "200nm shift")

#test_algorithm(cor_mirror_strip, cor_args, 0.7, "200nm shift")

# arguments for 10nm shift video
com_args_10nm = ["videos/10nm shift single.avi", 15, "bilinear", 3]
cor_args_10nm = [com_single, com_args_10nm, 8, 40, 2, 2]
grad_args_10nm = [com_single, com_args_10nm, 8, 40]
mp_args_10nm = [com_single, com_args_10nm, 8, 40, 2]


#test_algorithm(com_single, com_args_10nm, 0.07, "10nm shift")


# arguments for tethered videos
com_teth = ["videos/Tethered single.avi", 35]
grad_teth = [com_single, com_teth, 8, 40, "bilinear", 3]
cor_teth = [com_single, com_teth, 8, 40, 2, 2]
blob_teth = ["videos/Tethered single.avi", [120, 215, 0.8]]

#gen_train(blob, blob_teth)
#xcor, ycor = test_algorithm(cor_mirror_strip, cor_teth, 0.7, "Teth single", False, True)
#xblob, yblob = test_algorithm(blob, blob_teth, 0.7, "Teth single", False, True)
#xgrad, ygrad = test_algorithm(gradient_method, grad_teth, 0.7, "Teth single", False, True)


teth_lut = ["videos/Tethered LUT.avi", 30]
teth = ["videos/Tethered single.avi", 35]
lut_grad = [com_single, teth_lut, "grad", [8, 40, 2], teth_heights, True, "bilinear", 3]
teth_grad = [com_single, teth, "grad", [8, 40, 2], "bilinear", 3]
lut_rad = [com_single, teth_lut, "rad", [8, 40, 2], teth_heights]
#profs, z_lut = generate_LUT(*lut_rad)
#z = z_position("videos/Tethered LUT.avi", com_mirror, teth_lut, "rad", [profs, z_lut])


def f_wlc(Lext, F, Lp, kb, T, Lc):
    """Returns the force as predicted by the worm-like chain model
    Inputs:
        Lext - Tether extension length
        F - the force applied 
        Lp - persistence length
        kb - Boltzmann constant
        T - Temperature (K)
        Lc - contour length"""
    return 0.25 * (1 - (Lext/Lc))**(-2) - 0.25 + Lext/Lc - (F*Lp)/(kb * T)

def predict_fluct(x, y, z):
    """A function to predict the fluctuations at a given force and compare them to the obtained 
    fluctuations at the same force
    Inputs:
        x - array of x positions
        y - array of y positions
        z - array of z positions
    Outputs:
        pred_var - array of x,y,z predicted fluctuations
        variances - array of x,y,z obtained fluctuations"""
    steps = [2400, 3400, 4400, 5400, 6400, 7400, 8400, 9400]
    forces = [0.74, 1.44, 2.81, 5.49, 10.74, 15.01, 20.98, 29.33]
    variances = np.empty((8, 3))
    pred_var = np.empty((8, 3))
    Lc = 1020*10**(-9)
    Lp = 50*10**(-9)
    R = 1.4*10**(-6)

    zero_length = np.min(z)

    for i in range(len(steps)):
        step_x = x[steps[i]: steps[i] + 400]
        step_y = y[steps[i]: steps[i] + 400]
        step_z = z[steps[i]: steps[i] + 400]
        act = temph[steps[i]: steps[i] + 400]

        std_x = np.std(step_x * 142)
        std_y = np.std(step_y * 142)
        std_z = np.std(step_z * 0.88) # refractive index factor 0.88
    
        var_x = (std_x*10**-9)**2
        var_y = (std_y*10**-9)**2
        var_z = (std_z*10**-6)**2
        
        variances[i] = var_x, var_y, var_z

        L_meas = (np.average(step_z) - zero_length) * 10**(-6)
        L_exts = sp.optimize.fsolve(f_wlc, np.average(act), args=(forces[i], Lp, Lc, 1.38*10**(-23), 294))
        
        L_ext = L_exts[0] * 10**(-6)
        

        x_fluct = ((1.38*10**(-23) * 294) * L_ext)/(forces[i]*10**(-12)) 
        y_fluct = ((1.38*10**(-23) * 294) * (L_ext + R)/(forces[i]*10**(-12)))
        kz = (1.38*10**(-23) * 294)/(2 * Lp*Lc) * (2 + (1 - L_ext/Lc)**(-3))
        g = 6 * np.pi * 10**-3 * R
        h = R + L_ext
        g_brenner = g/(1 - (9*R)/(8*h) + (R**3)/(2*h**3) - (57*R**4)/(100*h**4) + R**5/(5*h**5) + (7*R**11)/(200*h**11) - R**12/(25*h**12))
        
        z_fluct = (2 * 1.38*10**(-23) * 294)/(np.pi * kz) * np.arctan((g_brenner * np.pi)/(kz * 10**(-3)))

        #print(f"{steps[i]}: {L_ext} | {L_meas} | {L_exts} | {np.average(act)}")
        


        pred_var[i] = x_fluct, y_fluct, z_fluct

    return pred_var, variances



#brownian(x, y, z)
#predict_fluct(x, y, z)

teth_lut = ["videos/Tethered LUT.avi", 30],
teth = ["videos/Tethered single.avi", 35]

lut_cent = [com_single, teth_lut, "grad", [8, 40, 2], teth_heights]
lut_grad = [com_single, teth_lut, "grad", [8, 40, 2]]
#test_z(lut_cent, lut_grad, 0.5, "tethered LUT")

teth_grad = [com_single, teth, "grad", [8, 40, 2]]
#test_z(lut_cent, teth_grad, 0.2, "tethered single")

# testing grad method for z
lut_args_fix = ["videos/cropped z LUT.avi", 35]
grad_lut_args_fix = [com_single, lut_args_fix, 8, 40, "bicubic", 4]
blob_lut_args_fix = ["videos/cropped z LUT.avi", [120, 215, 0.8]]


lut_args_teth = ["videos/Tethered LUT.avi", 35]
grad_lut_args_teth = ["videos/Tethered LUT.avi", com_single, lut_args_teth, 8, 40, "bilinear", 4]
blob_lut_args_teth = ["videos/Tethered LUT.avi", [120, 215, 0.8]]
gen_args_teth = ["videos/Tethered LUT.avi", blob, blob_lut_args_teth, "rad", [8, 40, 2], teth_heights]




# radial LUT
#profs, z_lut = generate_LUT("videos/cropped z LUT.avi", gradient_method, grad_lut_args_fix, "rad", [8, 40], lut_heights)
# widths LUT
#fit = generate_LUT("videos/cropped z LUT.avi", com_single, lut_args_fix, "grad", [8, 40, 2], lut_heights, interpolation="bilinear", scale=3)

# get rad and grad positions on LUT video
#z_rad = z_position("videos/cropped z LUT.avi", gradient_method, grad_lut_args_fix, "rad", [profs, z_lut])
#z_grad = z_position("videos/cropped z LUT.avi", com_single, lut_args_fix, "grad", [fit, 8, 40, 2], interpolation="bilinear", scale=3)

#profs_teth, z_teth = generate_LUT(*gen_args_teth)
#z_pos = z_position("videos/Tethered LUT.avi", blob, blob_lut_args_teth, "rad", [profs_teth, z_teth])



# args for z 20nm shift videos
z_20nm_args = ["videos/z 20nm single.avi", 35]
z_20nm_grad = ["videos/z 20nm single.avi", com_single, z_20nm_args, 8, 40, "bilinear", 3]
#z_20nm_rad = z_position("videos/z 20nm single.avi", com_single, z_20nm_args, "rad", [profs, z_lut])
#z_20nm_grad = z_position("videos/z 20nm single.avi", com_single, z_20nm_args, "grad", [fit, 8, 40, 2], interpolation="bilinear", scale=3)

#z_pos_10nm, gaps = z_position(z_10nm2, z_10nm.get_frame_count(), com_test, z_args3, "meow", 8, 30, fit_grad, deg=2, interpolation="bicubic", scale=3)

com_polar = ["videos/200nm shift every 5s cropped.avi"]
cor_polar = ["videos/200nm shift every 5s cropped.avi", com_single, ["videos/200nm shift every 5s cropped.avi"], 8, 30, 2, 2, "bicubic", 3]
grad_polar = [com_single, ["videos/200nm shift every 5s cropped.avi"], 8, 40]
#transform("videos/200nm shift every 5s cropped.avi", xy_func=gradient_method, xy_args=grad_polar)


test_algorithm(transform, ["videos/200nm shift every 5s cropped.avi", com_single, com_polar], 0.7, "200nm shift")