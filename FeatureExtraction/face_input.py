#encoding=utf-8
import numpy as np
import cv2
import os

def get_Trains_and_Lables(images_path,num_classes=None,is_gray=True,is_normal=True):
	trains,labels,names = [],[],[]
	labels_path = os.listdir(images_path)
	if num_classes:
		label_length = num_classes
	else:label_length = len(labels_path)
	labels_path.sort()
	num = 0
	for lable_name in labels_path:
		lable_path = os.path.join(images_path,lable_name)
		if os.path.isdir(lable_path):
			img_names = os.listdir(lable_path)
			img_names.sort();
			for img_name in img_names:
				# (1).读取图片
				img_path = os.path.join(lable_path,img_name)
				if is_gray:trains.append(cv2.imread(img_path,cv2.IMREAD_GRAYSCALE))
				else:trains.append(cv2.imread(img_path))
				# (2).标签
				temp_label = np.zeros(label_length)
				temp_label[num]=1
				labels.append(temp_label)
				# (3).名称
				names.append(lable_name)
			if num >=label_length-1:break;
			num+=1
	trains,labels = np.asarray(trains,dtype=np.float32),np.asarray(labels,dtype=np.float32)
	if is_gray:trains = trains.reshape(trains.shape[0],trains.shape[1],trains.shape[2],1)
	if is_normal:trains = trains/255.
	return trains,labels,names


if __name__ == '__main__':
	trains,labels,_ = get_Trains_and_Lables('../FaceAugment/tempImages')
	print(trains.shape)
	print(labels.shape)

