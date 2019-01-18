__author__ = 'Iacopo'
import renderer
import facial_feature_detector as feature_detection
import camera_calibration as calib
import scipy.io as io
import cv2
import numpy as np
import os
import check_resources as check
import matplotlib.pyplot as plt
import sys
import myutil
import ThreeD_Model
import config

this_path = os.path.dirname(os.path.abspath(__file__))
opts = config.parse()
## 3D Models we are gonna use to to the rendering {0, -40, -75}
newModels = opts.getboolean('renderer', 'newRenderedViews')
if opts.getboolean('renderer', 'newRenderedViews'):
    pose_models_folder = '/models3d_new/'
    pose_models = ['model3D_aug_-00_00', 'model3D_aug_-22_00', 'model3D_aug_-40_00', 'model3D_aug_-55_00',
                   'model3D_aug_-75_00']
else:
    pose_models_folder = '/models3d/'
    pose_models = ['model3D_aug_-00', 'model3D_aug_-40', 'model3D_aug_-75', ]
## In case we want to crop the final image for each pose specified above/
## Each bbox should be [tlx,tly,brx,bry]
resizeCNN = opts.getboolean('general', 'resizeCNN')
cnnSize = opts.getint('general', 'cnnSize')
if not opts.getboolean('general', 'resnetON'):
    crop_models = [None, None, None, None, None]  # <-- with this no crop is done.
else:
    # In case we want to produce images for ResNet
    resizeCNN = False  # We can decide to resize it later using the CNN software or now here.
    ## The images produced without resizing could be useful to provide a reference system for in-plane alignment
    cnnSize = 224
    crop_models = [[23, 0, 23 + 125, 160], [0, 0, 210, 230], [0, 0, 210, 230]]  # <-- best crop for ResNet
