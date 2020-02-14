#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
validate detection system
"""

import numpy as np

correctDetected, trueSigns, detectedSigns = 0.0,0,0  
signs = []
predictListSigns = []


#count number of correct classified signs, classified signs, true signs for recall and precision
#create list of predicted signs and a list of true signs for confusion matrix
def valid(predictList, num):
    
    global correctDetected, trueSigns, detectedSigns
    predictList = np.delete(predictList, np.where(predictList == 0))
    trueSigns +=1
    detect = np.argwhere(predictList == num)
    detectedSigns += len(detect)
    
    
    if len(predictList) == 0:
        signs.append(num)
        predictListSigns.append(0)
    else:
        if np.isin(num, predictList):
            correctDetected +=1
            signs.append(num)
            predictListSigns.append(num)
            index = np.argwhere(predictList == num)
            predictList = np.delete(predictList,index[0]) 
        else:
            signs.append(num)
            predictListSigns.append(predictList[0]) 
            predictList = np.delete(predictList,[0])

    if len(predictList) >= 1:
        for i in range(len(predictList)):
            signs.append(0)
            predictListSigns.append(predictList[i])



def return_validation():
    return correctDetected, detectedSigns, trueSigns, predictListSigns, signs
    
    
