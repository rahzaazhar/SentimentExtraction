#! /usr/bin/env python3
# -*-coding: utf-8-*-

import numpy as np
import cv2
import argparse
import pandas as pd
import random
import os
mapping1 = {'PrivateTest':'test', 'Training':'train', 'PublicTest':'validate'}
mapping2 = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Sad', 5:'Surprise', 6:'Neutral'}
curdir = os.path.join(os.path.abspath(os.path.dirname(__file__)),'data')
if not os.path.exists(curdir):
    os.mkdir(curdir)
def gen_record(csvfile,channel):
    data = pd.read_csv(csvfile,delimiter=',',dtype='a')
    labels = np.array(data['emotion'],np.float)
    # print(labels,'\n',data['emotion'])
        
    imagebuffer = np.array(data['pixels'])
    images = np.array([np.fromstring(image,np.uint8,sep=' ') for image in imagebuffer])
    del imagebuffer
    num_shape = int(np.sqrt(images.shape[-1]))
    images.shape = (images.shape[0],num_shape,num_shape)
    # img=images[0];cv2.imshow('test',img);cv2.waitKey(0);cv2.destroyAllWindow();exit()
    dirs = set(data['Usage'])
    subdirs = set(labels)
    class_dir = {}
    for dr in dirs:
        dest = os.path.join(curdir,mapping1[dr])
        class_dir[dr] = dest
        if not os.path.exists(dest):
            os.mkdir(dest)
            
    data = zip(labels,images,data['Usage'])
    #print(class_dir)
    #print(mapping1)
    for d in data:
        #print(d[-1])
        destdir = os.path.join(class_dir[d[-1]],mapping2[int(d[0])])
        if not os.path.exists(destdir):
            os.mkdir(destdir)
        img = d[1]
        filepath = unique_name(destdir,mapping1[d[-1]])
        print('[^_^] Write image to %s' % filepath)
        if not filepath:
            continue
        sig = cv2.imwrite(filepath,img)
        if not sig:
            print('Error')
            exit(-1)


def unique_name(pardir,prefix,suffix='jpg'):
    filename = '{0}_{1}.{2}'.format(prefix,random.randint(1,10**8),suffix)
    filepath = os.path.join(pardir,filename)
    if not os.path.exists(filepath):
        return filepath
    unique_name(pardir,prefix,suffix)
    


if __name__ == '__main__':
    #filename = 'fer2013.csv'
    parser = argparse.ArgumentParser(description='image path')
    parser.add_argument('--csv_path', nargs='?', type=str,    
                        help='Path to fer2013 csv file')
    args = parser.parse_args()
    #filename = os.path.join(curdir,filename)
    gen_record(args.csv_path,1)