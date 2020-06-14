import keras
import tensorflow
import cv2
import numpy as np
from keras import applications
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout
from keras.models import Model,Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from os.path import join
from keras.callbacks import TensorBoard,EarlyStopping
import os
import numpy as np
import cv2
sentiment_model = Sequential()
# input: 32x32 images with 1 channels -> (48, 48, 3) tensors.
# this applies 32 convolution filters of size 3x3 each.
sentiment_model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48 ,3)))
sentiment_model.add(Conv2D(32, (3, 3), activation='relu'))
sentiment_model.add(MaxPooling2D(pool_size=(2, 2)))
sentiment_model.add(Dropout(0.25))

sentiment_model.add(Conv2D(64, (3, 3), activation='relu'))
sentiment_model.add(Conv2D(64, (3, 3), activation='relu'))
sentiment_model.add(MaxPooling2D(pool_size=(2, 2)))
sentiment_model.add(Dropout(0.25))

sentiment_model.add(Flatten())
sentiment_model.add(Dense(256, activation='relu'))
sentiment_model.add(Dropout(0.5))
sentiment_model.add(Dense(7, activation='softmax'))

sentiment_model.load_weights("saved_weights.h5")
maping = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Neutral', 5:'Sad', 6:'Surprise'}
#maping = {0:'ANGRY', 1:'FEAR', 2:'HAPPY', 3:'NEUTRAL', 4:'SAD', 5:'SURPRISE'}
# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
#eye_cascade = cv2.CascadeClassifier('C:/Users/Nikhil/Desktop/Test/haarcascade_eye.xml')
angry=cv2.imread('Emoji_Data/angry.jpg')
angry=cv2.resize(angry,(200,200),interpolation=cv2.INTER_LANCZOS4)
fear=cv2.imread('Emoji_Data/fear.jpg')
fear=cv2.resize(fear,(200,200),interpolation=cv2.INTER_LANCZOS4)
happy=cv2.imread('Emoji_Data/happy.jpg')
happy=cv2.resize(happy,(200,200),interpolation=cv2.INTER_LANCZOS4)
neutral=cv2.imread('Emoji_Data/neutral.jpg')
neutral=cv2.resize(neutral,(200,200),interpolation=cv2.INTER_LANCZOS4)
sad=cv2.imread('Emoji_Data/sad.jpg')
sad=cv2.resize(sad,(200,200),interpolation=cv2.INTER_LANCZOS4)
surprise=cv2.imread('Emoji_Data/surprise.jpg')
surprise=cv2.resize(surprise,(200,200),interpolation=cv2.INTER_LANCZOS4)
disgust=cv2.imread('Emoji_Data/disgust.jpg')
disgust=cv2.resize(disgust,(200,200),interpolation=cv2.INTER_LANCZOS4)
cap = cv2.VideoCapture(0)

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        width=48
        height=48
        dim=(width,height)
        sub_face =cv2.resize(roi_color,dim,interpolation=cv2.INTER_LANCZOS4)
        #print(np.shape(sub_face))
        #cv2.imshow('extracted',sub_face)
        sub_face=np.expand_dims(sub_face,axis=0)
        #print(np.shape(sub_face))
        pred = sentiment_model.predict(sub_face)
        prediction = np.argmax(pred)
        print(maping[prediction])
        print(prediction)
        emotions=[angry,disgust,fear,happy,neutral,sad,surprise]
        cv2.imshow('img',emotions[prediction])
        cv2.waitKey(300)
        break
        
    cv2.imshow('Video Input Feed',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()