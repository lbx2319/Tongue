# -*- coding: utf-8 -*-
"""
Created on Sun May 31 22:13:38 2020

@author: jason
"""
import numpy as np
import cv2
import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime

def mask_visualization(GT, Predict):
    '''    

    Parameters
    ----------
    GT : TYPE 512*512 numpy.ndarray
        
    Predict : TYPE 512*512 numpy.ndarray
         
    
    Returns
    -------
    inter : TYPE 512*512 numpy.ndarray
            0 is background
            2 and 4 means FT and TF
            5 is tongue
            
            green : tongue => background
            blue : background => tongue
    miou : TYPE float
        mean IOU of tongue

    '''    
    
    GT = GT.astype('int8')
    Predict = Predict.astype('int8')
    
    # 0603 new
    inter = GT+Predict 
    union = GT-Predict
    miou = round(np.sum(inter == 2)/(np.sum(inter == 1)+np.sum(inter == 2)),4)
    
    
    inter[inter == 0] = 0
    inter[inter == 1] = 3 
    inter[inter == 2] = 5    
    
    inter = inter + union   
    
    # 0603 end
    return inter,miou

def write_excel(filename,miou):
    
    excel_path = 'E:/tongue/Image/Mask/test/result.xlsx'
    df = pd.read_excel(excel_path,index_col=0)
    filenameList = df.檔名.values
    
    if filename not in filenameList:
       df.loc[len(filenameList),'檔名'] = filename
       filenameList.append(filename)
       
    i = list(filenameList).index(filename)
    
    date = str(datetime.now().month).zfill(2)+str(datetime.now().day).zfill(2)    
    
    date = '0603_95'
    
    df.loc[i,date] = miou
    df.to_excel(excel_path)
    
def mask2pic(msk):
    # mask = np.expand_dims(msk, 2)
    mask = np.stack((msk,)*3, axis=-1)
    mask = mask * 255
    
    print(mask.shape)
    
    return mask
    

    


