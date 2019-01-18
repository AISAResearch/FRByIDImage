from feature_extraction import get_feature_from_images,get_sess_model,get_feature_by_sess_model
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

def imgHandle(img_data,alaph=1.4,beta=0):
    img_copy = img_data.copy().astype(np.float32)
    for i in range(img_copy.shape[0]):
        img_copy[i] = img_copy[i]*alaph+beta
        img_copy[i][img_copy[i]>=255] = 255
    return img_copy.astype(np.int)

img_path = "/Users/apple/PycharmProjects/UsePython2/data_augment/FaceRecognitionByIDImage/Img/999/0.jpg"
t_img_path = "/Users/apple/PycharmProjects/UsePython2/data_augment/FaceRecognitionByIDImage/Img/15/0.jpg"
img_data = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
t_img_data = cv2.imread(t_img_path,cv2.IMREAD_GRAYSCALE)
h_img_data = imgHandle(t_img_data)
t_img_data = t_img_data


def dis(v1,v2):
  return np.sum((v1-v2)**2)**0.5 / v1.shape[0]

if __name__ == '__main__':
  v1 = get_feature_from_images(img_data.reshape(1,img_data.shape[0],img_data.shape[1],1))
  v2 = get_feature_from_images(h_img_data.reshape(1,h_img_data.shape[0],h_img_data.shape[1],1))
  v3 = get_feature_from_images(t_img_data.reshape(1,t_img_data.shape[0],t_img_data.shape[1],1))
  print v1,"\n",v2,"\n",v3
  print v1[0]-v2[0]
  print dis(v1,v2)