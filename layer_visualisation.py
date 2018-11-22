from keras.models import load_model
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
from time import time
from matplotlib import pyplot as plt

classifier=load_model('/home/azhar/projects/sentiment_Extract_image/ModelBN.h5')
layer_outputs = [layer.output for layer in classifier.layers[1:7]] # Extracts the outputs of the top 12 layers
activation_model = Model(inputs=classifier.input, outputs=layer_outputs) # Creates a model that will return these outputs, given the model input
activation_model.summary()
img_tensor = cv2.imread('test0.jpg')#replace with image extracted from live feed
img_tensor = cv2.resize(img_tensor,(48,48))
img_tensor=np.expand_dims(img_tensor,axis=0) 
print(np.shape(img_tensor))
activations = activation_model.predict(img_tensor)
images_per_row = 8
layer_names = []

for layer in classifier.layers[1:7]:
    layer_names.append(layer.name) # Names of the layers, so you can have them as part of your plot
    
for layer_name, layer_activation in zip(layer_names, activations): # Displays the feature maps
    n_features = layer_activation.shape[-1] # Number of features in the feature map
    size = layer_activation.shape[1] #The feature map has shape (1, size, size, n_features).
    n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
    display_grid = np.zeros((size * n_cols, images_per_row * size))
    for col in range(n_cols): # Tiles each filter into a big horizontal grid
        for row in range(images_per_row):
            channel_image = layer_activation[0, :, :,col * images_per_row + row]
            channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable
            #print(channel_image.std())
            #channel_image = channel_image/channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size,row * size : (row + 1) * size] = channel_image
            scale = 1./size
    plt.figure(figsize=(scale * display_grid.shape[1],scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    print(np.shape(display_grid))
    plt.imshow(display_grid, aspect='auto', cmap='gray')
    plt.show()
