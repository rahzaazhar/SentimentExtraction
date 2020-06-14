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
import argparse
import numpy as np
import cv2

parser = argparse.ArgumentParser(description='image path')
parser.add_argument('--data_path', nargs='?', type=str,    
                        help='Path to data folder')
args = parser.parse_args()
path = args.data_path#specify path to folder with train,validate and test folders 
maping = {0:'ANGRY', 1:'FEAR', 2:'HAPPY', 3:'NEUTRAL', 4:'SAD', 5:'SURPRISE'}



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


# Creating new model. 
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
datagen = ImageDataGenerator(rescale=1./255)
train_generator = datagen.flow_from_directory(join(path,"train"),target_size=(48, 48),batch_size=32,class_mode='categorical')
validate_generator = datagen.flow_from_directory(join(path,"validate"),target_size=(48, 48),batch_size=32,class_mode='categorical')
test_generator = datagen.flow_from_directory(join(path,"test"),target_size=(48, 48),batch_size=32,class_mode='categorical')

#to check if mapping is corect execute
print(train_generator.class_indices)
#checkpointer1 = TensorBoard(log_dir='/home/azhar/sentiment_Extract_image/tensor_log')#make a folder named tensorlog and place its path here 
#checkpointer =  EarlyStopping(monitor='loss',min_delta=0.01,patience=10,verbose=1,mode='min')
sentiment_model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
sentiment_model.load_weights("saved_weights.h5")#specify path to weights
#for training include below line
sentiment_model.fit_generator(train_generator,steps_per_epoch=940,epochs=10,verbose=1,validation_data=validate_generator,validation_steps=50)#callbacks=[checkpointer1,checkpointer])
score=sentiment_model.evaluate_generator(test_generator,steps=113,verbose=1)
print('accuracy of the model:',score[1]*100)
sentiment_model.save_weights("weights_7classes.h5")#path to save new weights

#predicting single image
#X = cv2.imread('test0.jpg')#replace with image extracted from live feed
#X = cv2.resize(X,(48,48))
#print(np.shape(X))
#X=np.expand_dims(X,axis=0) 
#print(np.shape(X))
#pred = sentiment_model.predict(X)
#prediction = np.argmax(pred)
#print(maping[prediction])
