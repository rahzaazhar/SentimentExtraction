import numpy as np 
import cv2
image = cv2.imread('test1.jpeg')
face_cascade = cv2.CascadeClassifier('/home/azhar/tensorflow/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_default.xml')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray',gray)
faces = face_cascade.detectMultiScale(gray,1.3,5)
i=0
for (x,y,w,h) in faces:
	cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
#cv2.imshow('dettected faces',image)
cv2.imwrite('detectedFace1.jpeg',image)
#cv2.waitKey()
#cv2.destroyAllWindows()	