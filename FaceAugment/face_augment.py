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
from parse_path import get_filelist

this_path = os.path.dirname(os.path.abspath(__file__))

opts = config.parse()
## 3D Models we are gonna use to to the rendering {0, -40, -75}
newModels = opts.getboolean('renderer', 'newRenderedViews')
if opts.getboolean('renderer', 'newRenderedViews'):
    pose_models_folder = '/models3d_new/'
    pose_models = ['model3D_aug_-00_00', 'model3D_aug_-22_00', 'model3D_aug_-40_00', 'model3D_aug_-55_00']
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

# input and output dir

def augment_by_dir(input_path,outputFolder):
    nSub = opts.getint('general', 'nTotSub')
    fileList, num_update = get_filelist(input_path,outputFolder)
    if num_update==0:return 
    # check.check_dlib_landmark_weights()
    allModels = myutil.preload(this_path, pose_models_folder, pose_models, nSub)

    for f in fileList:
        if '#' in f:  # skipping comments
            continue
        splitted = f.split(',')
        image_key = splitted[0]
        image_path = splitted[1]
        image_landmarks = splitted[2]
        img = cv2.imread(image_path, 1)

        # img resize
        height,widht,challel = img.shape
        scale_factor = 300./widht
        img = cv2.resize(img,(int(widht*scale_factor),int(height*scale_factor)),interpolation=cv2.INTER_CUBIC)

        if image_landmarks != "None":
            lmark = np.loadtxt(image_landmarks)
            lmarks = []
            lmarks.append(lmark)
        else:
            print '> Detecting landmarks'
            lmarks = feature_detection.get_landmarks(img, this_path)

        if len(lmarks) != 0:
            ## Copy back original image and flipping image in case we need
            ## This flipping is performed using all the model or all the poses
            ## To refine the estimation of yaw. Yaw can change from model to model...
            img_display = img.copy()
            img, lmarks, yaw = myutil.flipInCase(img, lmarks, allModels)
            listPose = myutil.decidePose(yaw, opts, newModels)
            ## Looping over the poses
            for poseId in listPose:
                posee = pose_models[poseId]
                ## Looping over the subjects
                for subj in range(1, nSub + 1):
                    pose = posee + '_' + str(subj).zfill(2) + '.mat'
                    print '> Looking at file: ' + image_path + ' with ' + pose
                    # load detections performed by dlib library on 3D model and Reference Image
                    print "> Using pose model in " + pose
                    ## Indexing the right model instead of loading it each time from memory.
                    model3D = allModels[pose]
                    eyemask = model3D.eyemask
                    # perform camera calibration according to the first face detected
                    proj_matrix, camera_matrix, rmat, tvec = calib.estimate_camera(model3D, lmarks[0])
                    ## We use eyemask only for frontal
                    if not myutil.isFrontal(pose):
                        eyemask = None
                    ##### Main part of the code: doing the rendering #############
                    rendered_raw, rendered_sym, face_proj, background_proj, temp_proj2_out_2, sym_weight = renderer.render(
                        img, proj_matrix, \
                        model3D.ref_U, eyemask, model3D.facemask, opts)
                    ########################################################

                    if myutil.isFrontal(pose):
                        rendered_raw = rendered_sym
                    ## Cropping if required by crop_models
                    rendered_raw = myutil.cropFunc(pose, rendered_raw, crop_models[poseId])
                    ## Resizing if required
                    if resizeCNN:
                        rendered_raw = cv2.resize(rendered_raw, (cnnSize, cnnSize), interpolation=cv2.INTER_CUBIC)
                    ## Saving if required
                    if opts.getboolean('general', 'saveON'):
                        subjFolder = outputFolder + '/' + image_key.split('_')[0]
                        myutil.mymkdir(subjFolder)
                        savingString = subjFolder + '/' + image_key + '_rendered_' + pose[8:-7] + '_' + str(subj).zfill(
                            2) + '.jpg'
                        cv2.imwrite(savingString, rendered_raw)

                    ## Plotting if required
                    if opts.getboolean('general', 'plotON'):
                        myutil.show(img_display, img, lmarks, rendered_raw, \
                                    face_proj, background_proj, temp_proj2_out_2, sym_weight)
        else:
            print '> Landmark not detected for this image...'

def getallModels():
    nSub = opts.getint('general', 'nTotSub')
    allModels = myutil.preload(this_path, pose_models_folder, pose_models, nSub)
    return allModels

def get_face(img,allModels,n_number=1):
    nSub = opts.getint('general', 'nTotSub')
    result_face = []
    i=0
    lmarks = feature_detection.get_landmarks(img, this_path)
    if len(lmarks) != 0:
        img_display = img.copy()
        img, lmarks, yaw = myutil.flipInCase(img, lmarks, allModels)
        listPose = myutil.decidePose(yaw, opts, newModels)
        for poseId in listPose:
            posee = pose_models[poseId]
            for subj in range(1, nSub + 1):
                if i>=n_number:
                    return np.asarray(result_face)
                else: i=i+1
                pose = posee + '_' + str(subj).zfill(2) + '.mat'
                model3D = allModels[pose]
                eyemask = model3D.eyemask
                # perform camera calibration according to the first face detected
                proj_matrix, camera_matrix, rmat, tvec = calib.estimate_camera(model3D, lmarks[0])
                rendered_raw, rendered_sym, face_proj, background_proj, temp_proj2_out_2, sym_weight = renderer.render(
                    img, proj_matrix, \
                    model3D.ref_U, eyemask, model3D.facemask, opts)
                # add
                rendered_raw = cv2.resize(rendered_raw, (cnnSize, cnnSize), interpolation=cv2.INTER_CUBIC)
                result_face.append(rendered_raw)
               
    else:
        print '> Landmark not detected for this image...'

    return np.asarray(result_face)

# my_img_path = "input/430821199901100030.jpg"
if __name__ == '__main__':
    get_face(cv2.imread('../Img/timg.jpeg'),getallModels(),n_number=10)
    pass
    # augment_by_dir(,)






