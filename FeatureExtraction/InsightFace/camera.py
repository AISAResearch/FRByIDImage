import cv2

cap = cv2.VideoCapture(0)

while(True):
	res,frame=cap.read()
	if res:
		cv2.imshow('img',frame[:500,390:890])
		cv2.waitKey(1)