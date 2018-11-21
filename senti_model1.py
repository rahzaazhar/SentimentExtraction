import keras
import tensorflow
import cv2
import numpy as np
from keras import applications
from keras.models import load_model
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout,ReLU,Flatten,Input
from keras.layers import BatchNormalization
from keras.models import Model,Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from os.path import join
from keras.callbacks import TensorBoard,EarlyStopping
import os

path = "/home/azhar/projects/sentiment_Extract_image/train_data"#path to folder with train,validate and test folders 
maping = {0:'ANGRY', 1:'FEAR', 2:'HAPPY', 3:'NEUTRAL', 4:'SAD', 5:'SURPRISE'}

#Conv-(SBN)-ReLU-(Dropout)-(Max-pool)]M
#odel = load_model('home/azhar/projects/sentiment_Extract_image/MODELS/sav_model2.h5')
#ayer_dict = dict([(layer.name, layer) for layer in model.layers])

# = layer_dict['dropout_2'].output
#or layer in model.layers[:7]:
#layer.trainable = False

Input = Input(shape=(48,48,3))
x = Conv2D(64, kernel_size=(3, 3))(Input)
x = BatchNormalization()(x)
x = ReLU()(x)
x = Dropout(0.25)(x)
x = MaxPooling2D(pool_size=(2,2))(x)

x = Conv2D(128,kernel_size=(5,5))(x)
x = BatchNormalization()(x)
x = ReLU()(x)
x = Dropout(0.25)(x)
x = MaxPooling2D(pool_size=(2,2))(x)

x = Conv2D(512,kernel_size=(3,3))(x)
x = BatchNormalization()(x)
x = ReLU()(x)
x = Dropout(0.25)(x)


x = Conv2D(512,kernel_size=(3,3))(x)
x = BatchNormalization()(x)
x = ReLU()(x)
x = Dropout(0.25)(x)
x = MaxPooling2D(pool_size=(2,2))(x)

x = Flatten()(x)
x = Dense(256)(x)
x = BatchNormalization()(x)
x = ReLU()(x)
x = Dropout(0.25)(x)

x = Dense(512)(x)
x = BatchNormalization()(x)
x = ReLU()(x)
x = Dropout(0.25)(x)
x = Dense(6, activation='softmax')(x)

emotion_model = Model(inputs=Input,outputs=x)
emotion_model.summary()






sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
datagen = ImageDataGenerator(rescale=1./255)
train_generator = datagen.flow_from_directory(join(path,"train"),target_size=(48, 48),batch_size=32,class_mode='categorical')
validate_generator = datagen.flow_from_directory(join(path,"validate"),target_size=(48, 48),batch_size=32,class_mode='categorical')
test_generator = datagen.flow_from_directory(join(path,"test"),target_size=(48, 48, 3),batch_size=32,class_mode='categorical')

checkpointer1 = TensorBoard(log_dir='/home/azhar/sentiment_Extract_image/tensor_log')#make a folder named tensorlog and place its path here 
checkpointer =  EarlyStopping(monitor='loss',min_delta=0.01,patience=10,verbose=1,mode='min')
emotion_model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
emotion_model.fit_generator(train_generator,steps_per_epoch=940,epochs=35,verbose=1,callbacks=[checkpointer,checkpointer1],validation_data=validate_generator,validation_steps=50)
score=emotion_model.evaluate_generator(test_generator,steps=113,verbose=1)
print('accuracy of the model:',score[1]*100)
emotion_model.save_weights("/home/azhar/projects/sentiment_Extract_image/weightsBN.h5")#path to weights
emotion_model.save('/home/azhar/projects/sentiment_Extract_image/ModelBN.h5')#path to saved model















