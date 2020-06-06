# -*- coding: utf-8 -*-
"""
Created on Sat May 30 12:39:03 2020

@author: jason
"""


import cv2
import numpy as np
from Data_prepare_func import *
import os



def gettrainingdata():
    path = 'E:/tongue/Image/Mask/Mask_Fix/(43).png'
    CorrectionImage_path = 'E:/tongue/Image/Mask/Color_Image'
    SourceImage_path = 'E:/tongue/Image/Src_Image'
    Mask_path = 'E:/tongue/Image/Mask/Mask_Fix'
    tar_path = 'E:/tongue/Image/Img'
    GT_path = 'E:/tongue/Image/Mask/GT'
    test_path = 'E:/tongue/Image/Mask/test'
    test2_path= 'E:/tongue/Image/Mask/test2'
    mobile_path= 'E:/tongue/Image/Mask/mobile'
    files = os.listdir(mobile_path)
    
    for f in files:
        # print(SourceImage_path+f'/{f[:-4]}.JPG')
        
        #mask = mask_resize2512(Mask_path+'/'+f) 
        #img_cor = resize2512(CorrectionImage_path+'/'+f) 
        img_cor = resize2512(mobile_path+'/'+f) 
        #img_sor = resize2512(SourceImage_path+f'/{f[:-4]}.JPG') 
        cv2.imwrite(test2_path+'/'+f,img_cor)
        #cv2.imwrite(test2_path+'/ori_'+f,img_sor)
        #cv2.imwrite(GT_path+'/'+f,mask)
        #cv2.imwrite(GT_path+'/ori_'+f,mask)
        
def getATDSMASK():
    test_path = 'E:/tongue/Image/Mask/test'
    ATDSMask_path = 'E:/tongue/Image/Mask/Mask_ATDS'
    
    files = os.listdir(test_path)
    for f in files:       
        img = cv2.imread(test_path+'/'+f,0)    
        
        ret,img = cv2.threshold(img,1,255,cv2.THRESH_BINARY)        
        
        cv2.imwrite(ATDSMask_path+'/'+f,img)

# getATDSMASK()
gettrainingdata()

    
