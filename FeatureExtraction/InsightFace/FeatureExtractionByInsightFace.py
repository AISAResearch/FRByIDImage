import face_embedding
import argparse
import cv2
import numpy as np
import os
import sys

this_path = os.path.dirname(os.path.abspath(__file__))
pre_path = os.path.abspath(os.path.join(this_path,".."))
sys.path.append(pre_path)
import face_input

parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default=os.path.join(this_path,'models/model-r34-amf/model,0'), help='path to load model.')
parser.add_argument('--cpu', default=1, type=int, help='gpu id')
parser.add_argument('--det', default=2, type=int, help='mtcnn option, 2 means using R+O, else using O')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
args = parser.parse_args()

def getModel():
	return face_embedding.FaceModel(args)

def getImageVector(img_path,model):
	img = cv2.imread(img_path)
	fature = model.get_feature(img)
	return np.asarray(fature).astype(np.float32)
	
def getImageVectorFromData(img_data,model):
	res = np.asarray(model.get_feature(img_data)).astype(np.float32)
	return res

def featureExtractionByImages(imgs):
	feature = []
	model = getModel()
	for img in imgs:
		fea = getImageVectorFromData(img,model)
		succ = True
		try:
		 fea.shape[0]==512
		except Exception as e:
			succ = False
			print e
		if succ:feature.append()
	return np.asarray(feature,dtype=np.float32)

def featureExtractionByImgPath(imgPath):
	images, labels , names = face_input.get_Trains_and_Lables(img_path)
	features = featureExtractionByImages(images)
	return features,labels,names

if __name__ == '__main__':
	model = getModel()
	img1 = getImageVectorFromData(cv2.imread('img/1.jpg'),model)
	print img1

