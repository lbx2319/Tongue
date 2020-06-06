# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2
import numpy as np

def resize2512(path):
    '''    

    Parameters
    ----------
    path : TYPE string 
        Target image path.

    Returns
    -------
    resize_img : numpy.ndarray
        completed resize to 512*512 image.

    '''
    img = cv2.imread(path)
    
    '''
    Get the width & height of image,then decide the value of padding
    '''
    size = img.shape
    height = size[0]
    width = size[1]  
    
    #phone image
    if height > width: 
        pad = int((height - width) / 2)
        pad_img = cv2.copyMakeBorder(img,0,0, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    #camera image
    else:
        pad = int((width - height) / 2)
    
        pad_img = cv2.copyMakeBorder(img,pad,pad, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    resize_img = cv2.resize(pad_img,(512 ,512))
    
    return resize_img


def mask_resize2512(path):
    '''
    

    Parameters
    ----------
    path : TYPE string
        Target mask path.

    Returns
    -------
    resize_img : numpy.ndarray
        completed resize to 512*512 mask, with one channel

    '''
    img = cv2.imread(path,0)
    
    '''
    Get the width & height of image,then decide the value of padding
    '''
    ret,img = cv2.threshold(img,254,255,cv2.THRESH_BINARY)
    # cv2.imwrite(f'E:/tongue/Image/Mask/test/test2.png',img)
    
    
    
    size = img.shape
    height = size[0]#1424
    width = size[1]#2136  
    
    # set the first row and col to 0
    for i in range(width-1):
        img[0,i] = 0
    
    for i in range(height-1):
        img[i,0] = 0
        
    #phone image
    if height > width: 
        pad = int((height - width) / 2)
    #camera image
    else:
        pad = int((width - height) / 2)
    
    pad_img = cv2.copyMakeBorder(img,pad,pad, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    resize_img = cv2.resize(pad_img,(512 ,512))
    resize_img = cv2.morphologyEx(resize_img,cv2.MORPH_OPEN,(3,3))
    resize_img = cv2.morphologyEx(resize_img,cv2.MORPH_CLOSE,(3,3))
    
    return resize_img



