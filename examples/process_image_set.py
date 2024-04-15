#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""

used to evaluate ST alg input params and their effect
on solution accuracy and solve times

"""

################################
#LOAD LIBRARIES
################################
import os
import cv2
import csv
import json
import time
import psutil
import subprocess
import numpy as np
import pandas as pd                                                             
from scipy.spatial.transform import Rotation as R  
from datetime import datetime
from star_tracker import main
from star_tracker.cam_matrix import *
from star_tracker.array_transformations import *

import io

cols = 1280
rows = 960
pixels = cols * rows
capture_file = ""

path="/dev/prucam"


# open up the prucam char device
fd = os.open(path, os.O_RDWR)
fio = io.FileIO(fd, closefd = False)

num_captures = 1
for x in range(num_captures):
    # make buffer to read into
    imgbuf = bytearray(pixels)

    # read from prucam into buffer
    fio.readinto(imgbuf)

    # read image bytes into ndarray
    img = np.frombuffer(imgbuf, dtype=np.uint8).reshape(rows, cols)

    # do bayer color conversion. For monochrome/raw image, comment out
    img = cv2.cvtColor(img, cv2.COLOR_BayerBG2BGR)

    # json encode image
    ok, img = cv2.imencode('.png', img)
    if not(ok):
        raise BaseException("encoding error")

    capture_file = ('capture' + str(x) + '.png')

    # write image
    with open(capture_file, 'wb') as f:
        f.write(img)

################################
#USER INPUT
################################
nmatch = 6 # minimum number of stars to match
starMatchPixelTol = 2 # pixel match tolerance
min_star_area = 5 # minimum pixel area for a star
max_star_area = 200 # maximum pixel area for a star
max_num_stars_to_process = 20 # maximum number of centroids to attempt to match per image

low_thresh_pxl_intensity = None
hi_thresh_pxl_intensity = None

VERBOSE = True # set True for prints on results
graphics = True # set True for graphics throughout the solve process


data_path = '/home/debian/oresat-star-tracker-software/oresat_star_tracker/cots-Star-Tracker-master/data/' # full path to your data
#image_path = '/Users/kyleklein/Desktop/psas/cots/py_src/tools/camera_calibration/tetra/img_data/' # full path to your images
image_path = '.'

cam_config_file_path = '/home/debian/oresat-star-tracker-software/oresat_star_tracker/cots-Star-Tracker-master/data/cam_config/generic_cam_params.json' # full path (including filename) of your cam config file
darkframe_file_path = '/home/debian/oresat-star-tracker-software/oresat_star_tracker/cots-Star-Tracker-master/py_src/tools/camera_calibration/tetra/img_data/autogen_darkframe.jpg' # full path (including filename) of your darkframe file
image_extension = ".png" # the image extension to search for in the data_path directory
cat_prefix ='' # if the catalog has a prefix, define it here

################################
#SUPPORT FUNCTIONS
################################
def convert_quaternion(sequence, x, y, z, w, degrees):                      
    r = R.from_quat([x,y,z,w])                                              
    r_q = r.as_quat()                                                       
    r_m = r.as_matrix()                                                     
    r_e = r.as_euler(sequence,degrees)                                      
    return r_q, r_m, r_e  

################################
#MAIN CODE
################################
#load star tracker stuff
if darkframe_file_path == '': darkframe_file_path = None
if darkframe_file_path is not None:
    if not os.path.exists(darkframe_file_path):
        darkframe_file_path = None
        print("unable to find provided darkframe file, proceeding without one...")
    else:    print("darkframe file: " + darkframe_file_path)
else:    print("no darkframe file provided, proceeding without one...")

k = np.load(os.path.join(data_path, cat_prefix+'k.npy'))
m = np.load(os.path.join(data_path, cat_prefix+'m.npy'))
q = np.load(os.path.join(data_path, cat_prefix+'q.npy'))
x_cat = np.load(os.path.join(data_path, cat_prefix+'u.npy'))
indexed_star_pairs = np.load(os.path.join(data_path, cat_prefix+'indexed_star_pairs.npy'))

cam_file = cam_config_file_path
camera_matrix, _, _ = read_cam_json(cam_file)
dx = camera_matrix[0, 0]
isa_thresh = starMatchPixelTol*(1/dx)

#define structures for data capture
image_name = []
ttime = []
stemp = []
sram  = []
scpu  = []
solve_time = []
qs = []
qv0 = []
qv1 = []
qv2 = []

image_names = []

# create list of all images in target dir
#---------------------------------------
#total_start = time.time()
#dir_contents = os.listdir(image_path)

#for item in dir_contents:
#    if image_extension in item:
#        image_names+=[os.path.join(os.path.abspath(image_path),item)]

#for image_filename in image_names:

#    image_name += [image_filename]
#    print("===================================================")
#    print(image_filename)
#----------------------------------------

# single straight from capture file
image_filename = capture_file

#run star tracker
solve_start_time = time.time()

q_est, idmatch, nmatches, x_obs, rtrnd_img = main.star_tracker(
        image_filename, cam_file, m=m, q=q, x_cat=x_cat, k=k, indexed_star_pairs=indexed_star_pairs, darkframe_file=darkframe_file_path, 
        min_star_area=min_star_area, max_star_area=max_star_area, isa_thresh=isa_thresh, nmatch=nmatch, n_stars=max_num_stars_to_process,
        low_thresh_pxl_intensity=low_thresh_pxl_intensity,hi_thresh_pxl_intensity=hi_thresh_pxl_intensity,graphics=graphics,verbose=VERBOSE)

# remove captured image after processing
os.remove(image_filename)

solve_time += [time.time()-solve_start_time]

# collect data
try:
    assert not np.any(np.isnan(q_est))
    if VERBOSE:
        print('est q: ' + str(q_est)+'\n')
    qs += [q_est[3]]
    qv0 += [q_est[0]]
    qv1 += [q_est[1]]
    qv2 += [q_est[2]]
except AssertionError:
    if VERBOSE:
        print('NO VALID STARS FOUND\n')
    qs += [999]
    qv0 += [999]
    qv1 += [999]
    qv2 += [999]

# get quaternions for one file
the_length = len(qs)
q_matrix_x = qv0
q_matrix_y = qv1
q_matrix_z = qv2
q_matrix_w = qs

euler_matrix = np.zeros([the_length,3])

# convert quaternions to attitude info
for i in range(the_length):

    if (q_matrix_x[i] == 999) or (q_matrix_y[i] == 999) or (q_matrix_z[i] == 999) or (q_matrix_w[i] == 999):
        euler_matrix[i,:] = np.array([0,0,0])

    else:
        euler_matrix[i] = convert_quaternion('ZXZ', q_matrix_x[i], q_matrix_y[i], q_matrix_z[i], q_matrix_w[i], degrees=True)[2]
        euler_matrix[i,0] = euler_matrix[i,0] - 90
        euler_matrix[i,1] = 90 - euler_matrix[i,1]
        euler_matrix[i,2] = euler_matrix[i,2] + 180

# return ra, dec, roll
#return euler_matrix[:,0], euler_matrix[:,1], euler_matrix[:,2]

###############################################################################
# Stores data in a pandas data frame and saves it in an excel file

# single captured image
df = pd.DataFrame((image_names,q_matrix_x,q_matrix_y,q_matrix_z ,q_matrix_w ,euler_matrix[:,0],euler_matrix[:,1],euler_matrix[:,2]),('File','Q,x','Q,y','Q,z','Q,w','ra3','dec1','roll3'),(np.linspace(1,1,1))).T

# image(s) from disk
#df = pd.DataFrame((image_names,q_matrix_x,q_matrix_y,q_matrix_z ,q_matrix_w ,euler_matrix[:,0],euler_matrix[:,1],euler_matrix[:,2]),('File','Q,x','Q,y','Q,z','Q,w','ra3','dec1','roll3'),(np.linspace(1,len(image_names),len(image_names)))).T

df.to_csv(image_path+"_converted.csv")

