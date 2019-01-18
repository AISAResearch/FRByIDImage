#encoding=utf-8
import os
import cv2
import numpy as np
import sys
import tensorflow as tf
import time
import math
import random

this_path = os.path.dirname(os.path.abspath(__file__))

# 1.人脸扩充的文件夹
FaceAugment_dir = os.path.join(this_path,"FaceAugment")
sys.path.append(FaceAugment_dir)
import face_augment

# 2.人脸特征提取的文件夹
FeatureExtraction_base_dir = os.path.join(this_path,"FeatureExtraction")
sys.path.append(FeatureExtraction_base_dir)
import face_input

# 2.1利用Resnet提取特征的文件夹
FaceExtraction_Resnet_dir = os.path.join(FeatureExtraction_base_dir,"ResNet")
sys.path.append(FaceExtraction_Resnet_dir)
from feature_extraction import get_feature_from_imgPath,get_feature_from_images

# 2.2利用insightface里面提取特征的文件夹
FaceExtraction_InsightFace_dir = os.path.join(FeatureExtraction_base_dir,"InsightFace")
sys.path.append(FaceExtraction_InsightFace_dir)
from FeatureExtractionByInsightFace import getModel,getImageVectorFromData,featureExtractionByImages

input_dir = "InputImage"
temp_dir = os.path.join(FaceAugment_dir,'tempImages')

def get_face_vec(input_dir,temp_dir,method=get_feature_from_images):
	# 1.首先扩充人脸
	face_augment.augment_by_dir(input_dir,temp_dir)
	# 2.进行特征提取
	images, labels , names = face_input.get_Trains_and_Lables(temp_dir,is_normal=True,is_gray=False)
	start = time.time()
	features = method(images,)
	print "将图片嵌入时间",time.time() - start,'秒'
	print names
	return features,labels,names

def countV(values):
    sing_v = np.unique(np.array(values))
    counts = np.zeros(sing_v.shape)
    for idx,v in enumerate(sing_v):
        counts[idx] = np.sum(values==v)
    print(sing_v)
    print(counts)
    max_idx = np.argmax(counts)
    print(max_idx)
    return max_idx,sing_v[max_idx]

def knn(features,names,input_img,k=10):
	n_face = features.shape[0]
	img_ = np.tile(input_img,np.array([n_face,1]))
	distances = (features - img_)**2
	distances = distances.sum(axis=1)**0.5
	min_indexes = np.argsort(distances)[:k]
	res = [names[index] for index in min_indexes]
	print np.sort(distances)[:k]
	return res

def cos_cal(v1,v2):
    return np.sum(v1*v2) / ( (np.sum(v1**2)**0.5) * (np.sum(v2**2)**0.5))

def cos_knn(features,names,input_img,k=10):
	n_face = features.shape[0]
	distances = [cos_cal(feature,input_img) for feature in features]
	distances = np.asarray(distances,dtype=np.float32)
	print distances
	max_indexes = np.argsort(distances)[-k:]
	res = [names[index] for index in max_indexes]
	print np.sort(distances)[-k:]
	return res

def temp_knn(features,names,input_imgs,k=12,knn_method=knn):
	res = []
	batch_size = features.shape[0]
	for i in range(k):
		indexes = [index for index in range(batch_size) if index%k==i]
		res.append(knn_method(features[indexes],[names[index] for index in indexes],input_imgs[i],k=1)[0])
	print countV(res)
	return res

def imgHandle(img_data,alaph=1.4,beta=0,is_normal=False):
    img_copy = img_data.copy().astype(np.float32)
    for i in range(img_copy.shape[0]):
        img_copy[i] = img_copy[i]*alaph+beta
        if is_normal:img_copy[i][img_copy[i]>=1] = 1
        else:img_copy[i][img_copy[i]>=255] = 255
    return img_copy.astype(np.int)

def get_camera(method=get_feature_from_images):
	# features,labels,names = get_face_vec(input_dir,temp_dir,method)
	allModels = face_augment.getallModels()

	if method==featureExtractionByImages:
		model = getModel()
		pass
	cap = cv2.VideoCapture(0)
	j = 20
	while True:
		ret,frame = cap.read()
		if ret:
			try:
				my_path = "Img/"+str(j)
				os.mkdir(my_path)
				j+=1
				# 将人脸扩充,然后取扩充后的第一张图片 并将其转换成 灰度图
				frame_ = cv2.resize(frame[:630,285:795],(102,126),interpolation=cv2.INTER_CUBIC)
				res = face_augment.get_face(frame_,allModels,n_number=12)
				i=0
				for r in res:
					cv2.imwrite(my_path+"/"+str(i)+".jpg",r)
					i+=1
				cv2.imshow('current',frame[:630,285:795])
				
				# # (1).单独一张进行测试
				# gray_res = cv2.cvtColor(res[0],cv2.COLOR_BGR2GRAY)
				# input_img = gray_res.reshape(1,gray_res.shape[0],gray_res.shape[1],1)/255.
				# vec = method(input_img)
				# print knn(features,names,vec[0])
				# cv2.imshow('now',gray_res)
				# cv2.waitKey(1)

				# # (2).扩充到的全部数据进行测试

				# 转灰度图 并且调整亮度
				# gray_res = [imgHandle(cv2.cvtColor(res[i],cv2.COLOR_BGR2GRAY)) for i in range(res.shape[0])]
				# gray_res = res

				# gray_res = np.asarray(gray_res,dtype=np.float32)/255.
				# veces = method(gray_res.reshape(gray_res.shape[0],gray_res.shape[1],gray_res.shape[2],-1))
				# if(veces.shape[0]==12):print temp_knn(features,names,veces,knn_method=knn)
				# for i in range(12):
				# 	cv2.imwrite(my_path+"/"+str(i)+".jpg",gray_res[i])
				cv2.waitKey(100)

			except Exception as e :
				print e
	cap.release()
	return

def fromImages(img_paths,method=get_feature_from_images):

	features,labels,names = get_face_vec(input_dir,temp_dir,method)
	allModels = face_augment.getallModels()

	if method==featureExtractionByImages:
		model = getModel()
		pass
	while True:
		for img_name in os.listdir(img_paths):
			img_path = os.path.join(img_paths,img_name)
			frame = cv2.imread(img_path)
			try:
				# 将人脸扩充,然后取扩充后的第一张图片 并将其转换成 灰度图
				res = face_augment.get_face(frame,allModels,n_number=12)
				print "res shape",res.shape
				# # (1).单独一张进行测试
				# gray_res = cv2.cvtColor(res[0],cv2.COLOR_BGR2GRAY)
				# input_img = gray_res.reshape(1,gray_res.shape[0],gray_res.shape[1],1)/255.
				# vec = method(input_img)
				# print knn(features,names,vec[0])
				# cv2.imshow('now',gray_res)
				# cv2.waitKey(1)

				# # (2).扩充到的全部数据进行测试
				gray_res = [cv2.cvtColor(res[i],cv2.COLOR_BGR2GRAY)/255. for i in range(res.shape[0])]
				gray_res = np.asarray(gray_res,dtype=np.float32)
				veces = method(gray_res.reshape(gray_res.shape[0],gray_res.shape[1],gray_res.shape[2],1))
				if(veces.shape[0]==12):print temp_knn(features,names,veces)
				cv2.imshow('now',gray_res[0])
				cv2.waitKey(1)

			except Exception as e :
					print e
			time.sleep(5)

def o_knn(all_,single,names):
	temp = np.tile(single,np.array([all_.shape[0],1]))
	distances = all_ - temp
	distances = np.sum(distances**2,axis=1)**0.5
	print distances
	idx = np.argmin(distances)
	res = -1
	if np.isnan(distances[idx]):print '未检测到人脸'
	elif distances[idx]>1:
		print "该人脸不在数据集中"
		res = 0
	else:
		print '检测到结果',names[idx]
		res = 1
	return res,names[idx]

def rotate(img,angle):
    height = img.shape[0]
    width = img.shape[1]

    if angle%180 == 0:
        scale = 1
    elif angle%90 == 0:
        scale = float(max(height, width))/min(height, width)
    else:
        scale = math.sqrt(pow(height,2)+pow(width,2))/min(height, width)
    rotateMat = cv2.getRotationMatrix2D((width/2, height/2), angle, scale)
    rotateImg = cv2.warpAffine(img, rotateMat, (int(width/1),int(height/1)))
    return rotateImg

def tfactor(img):
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV);  #增加饱和度光照的噪声
    hsv[:,:,0] = hsv[:,:,0]*(0.8+ np.random.random()*0.2);
    hsv[:,:,1] = hsv[:,:,1]*(0.3+ np.random.random()*0.7);
    hsv[:,:,2] = hsv[:,:,2]*(0.5+ np.random.random()*0.5);
    img = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR);
    return img

def r():
	return 1 if random.random()<0.5 else -1

def img_simple_aug(src_path,dir_path,angle_v=10,num_add=3):
	i=0
	for img_name in os.listdir(src_path):
		if img_name.endswith(".jpg"):
			img_path = os.path.join(src_path,img_name)
			img_data = cv2.imread(img_path)
			j=0
			while j<num_add:
				j+=1
				img_copy = img_data.copy()
				img_copy = rotate(tfactor(img_copy),int(random.random()*angle_v*r()))
				name = dir_path+"/1_"+str(j)+"_"+img_name
				print name
				cv2.imwrite(name,img_copy)
			i+=1
	print "总共扩充了",i,"张图片,每张图片扩充了",num_add,"张"


def other(input_img_path="InputImage"):
	# (1).将身份证图片转换成特征向量
	model = getModel()
	all_names = []
	all_features = []
	start = time.time()
	num_img = 0
	for img_name in os.listdir(input_img_path):
		if img_name.endswith("jpg"):
			img_path = os.path.join(input_img_path,img_name)
			all_names.append(img_name)
			all_features.append(getImageVectorFromData(cv2.imread(img_path),model))
			num_img+=1
	all_features = np.asarray(all_features,dtype=np.float32)
	
	print all_names
	temp_time = time.time()
	print "总共提取了",num_img,"张人脸,总耗时",temp_time -  start
	if num_img!=0:print "平均每人耗时",(temp_time - start)/num_img

	# (2).开启摄像头
	all_camera_time,num_camera,all_dec,num_dec=0,0,0,0

	labels,results = [],[]
	for img_name in os.listdir("images"):
		if img_name.endswith("jpg"):
			img_path = os.path.join("images",img_name)
			frame = cv2.imread(img_path)

			labels.append(int(img_name[0]))

			# temp_time = time.time()
			# all_camera_time += (time.time()-temp_time)
			num_camera+=1
			# print "非累积时间:",time.time()-temp_time
			# print "摄像头的读取一张人的时间:",all_camera_time/num_camera,"则速度为",num_camera/all_camera_time,"张/秒"

			temp_time = time.time()
			
			# frame = cv2.resize(frame[:630,285:795],(102,126),interpolation=cv2.INTER_CUBIC)
			try:
				res = getImageVectorFromData(frame,model)
				val,im =  o_knn(all_features,res,all_names)
				all_dec+=(time.time()-temp_time)
				num_dec+=1
				print "提取人脸特征并且判断时间:",all_dec/num_dec,"则速度为",num_dec/all_dec,"张/秒"

				if val == 1:
					if im != img_name[-len(im):]:val=-2
				results.append(val)
			except Exception as e:
				print e
			print "平均处理一次的时间:",(all_camera_time+all_dec)/num_camera
			print "则速度为",num_camera/(all_camera_time+all_dec)
			print ""
			# cv2.imshow("window",frame)
			# cv2.waitKey(1)

	print results 
	print labels
			

"""
source activate py2
"""

if __name__ == "__main__":
	# fromImages('img')
	# get_camera()
	other()
	# img_simple_aug("InputImage","images")
	pass

