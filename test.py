# -*- coding: utf-8 -*-
"""
Created on Sat May 30 19:11:51 2020

@author: jason
"""


import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import os
import cv2
from keras.models import load_model
from keras import backend as K
from PIL import Image
from test_func import mask_visualization,write_excel,mask2pic


def mean_iou(y_true, y_pred):
    yt0 = y_true[:,:,:,0]
    yp0 = K.cast(y_pred[:,:,:,0] > 0.5, 'float32')
    inter = tf.math.count_nonzero(tf.logical_and(tf.equal(yt0, 1), tf.equal(yp0, 1)))
    union = tf.math.count_nonzero(tf.add(yt0, yp0))
    iou = tf.where(tf.equal(union, 0), 1., tf.cast(inter/union, 'float32'))
    return iou

#loading model
model = load_model('unet_0530.h5',custom_objects={"mean_iou": mean_iou})
# model.summary()

# testing Image
test_path = 'E:/tongue/Image/Mask/test'
test2_path = 'E:/tongue/Image/Mask/test2'
GT_path = 'E:/tongue/Image/Mask/GT'
tar_path = 'E:/tongue/Image/Img'
mobile_path = 'E:/tongue/Image/Mask/mobile'

files = os.listdir(mobile_path)
index = 1

for f in files:
    print(index,'/',len(files))
    index+=1
    raw = Image.open(mobile_path+'/'+f)
    #raw = Image.open(tar_path+'/'+f)
    raw = np.array(raw)/255.
    raw = raw[:,:,0:3]

#predict the mask 
    pred = model.predict(np.expand_dims(raw, 0))

#mask post-processing 
    msk  = pred.squeeze()
    # msk = np.stack((msk,)*3, axis=-1)
    msk[msk >= 0.5] = 1
    msk[msk < 0.5] = 0 
    msk = np.array(msk)
    msk = msk.astype('uint8')
    
    msk = mask2pic(msk)
    
    msk_cv = cv2.cvtColor(msk,cv2.COLOR_RGB2BGR)
    img_cv = cv2.imread(mobile_path+'/'+f)
    
    dst=cv2.addWeighted(img_cv,0.7,msk_cv,0.3,0)
    
    cv2.imwrite(mobile_path+'/Perdict_'+f,dst)
    
    # GT = cv2.imread(GT_path+'/'+f,0) 
    # print(GT.shape)
    # print(msk.shape)
    # Predict = msk[:,:,0]
    #GT = np.array(GT)/255
    # C,miou = mask_visualization(GT,Predict)  
    #C,miou = mask_visualization(GT,msk)
    
    
    #plt.figure(figsize=(5.12,5.12),dpi = 100)
    #plt.axis('off')
    #plt.imshow(msk)
    #plt.savefig(mobile_path+'/Perdict_'+f,bbox_inches='tight',dpi=100)
    #plt.show()
    #write_excel(f[:-4], miou)    