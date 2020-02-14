#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
traffic sign detection system for model cars 
using support vector machines and the histogram of oriented gradients
"""

import cv2
import os
import image_process
import numpy as np
import matplotlib.pyplot as plt
import time
from validate import valid, return_validation
from sklearn.metrics import confusion_matrix

categories = ["nosign", "30", "50", "60", "stop", "priority", "yield" ]
dic_shapes = { "nosign": 0, "30": 1, "50": 2, "60": 3, "stop": 4, "priority": 5, "yield": 6}



svm1 = "/home/rima/Dokumente/BA/github/final_version/model/svm1.xml"
svm2 = "/home/rima/Dokumente/BA/github/final_version/model/svm_number.xml"

def detect(rgb):
#   for validation
    predictList = []            
    featureList = []
    index = 0

    rgb = cv2.resize(rgb,(400,300))
    
    gray = cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    
    #red, yellow segmentation
    segIm = image_process.color_segmentation(rgb)
    
#   filtering ROIs
    statsListR,roiListR = image_process.get_roi(segIm[0],rgb, 1)
    
    statsListY,roiListY = image_process.get_roi(segIm[1],rgb, 2)
    
    statsList = statsListR + statsListY
    roiList = roiListR + roiListY
    
  
##  HOG feature of shape
    if len(roiList) > 0:
        for idx, roi in enumerate(roiList):

            feature = image_process.hogFeature(roi,(60,60), 9,(8,8), (2,2), 'L2')

            if feature is not None:  
                featureList.append(feature)

#   predict shape and sign        
    if len(featureList) > 0:    

        featureList = np.float32(featureList)

        predictList = svm1.predict(featureList)[1].ravel()

        #speed limit sign -> predict number
        for idx, predict in enumerate(predictList):
            if predict == 1 or predict == 2 or predict == 3:
                feat= []

                img = np.uint8(gray[statsList[idx][1]:statsList[idx][3], statsList[idx][0]:statsList[idx][2]])

                img = cv2.equalizeHist(img)
                numIm = image_process.hogFeature(img, (100,100),9,(6,6),(2,2),'L2')

                feat.append(numIm)
                predictNum = svm_num.predict(np.float32(feat))[1].ravel()
                predictList[idx]= predictNum

#       draw bounding box 
        for index,predict in enumerate(predictList):
            if predict > 0:
                drawBoundingBox(statsList[index], predictList[index],rgb)
      
        print("predicted signs: %s" % predictList)
        
#   validation     
    num = dic_shapes[category]
    valid(predictList, num)
        





             
def confusion_mat():
    
    cm = confusion_matrix(signs,predictListSigns, labels = [0,1,2,3,4,5,6])
    print(cm)
                    
def drawBoundingBox(roi, trafficSign, rgb):
    
    sign = int(trafficSign)
    
    cv2.rectangle(rgb,(roi[0],roi[1]),(roi[2],roi[3]),(0,0,255),2)
    cv2.putText(rgb,categories[sign],(roi[0],roi[3]+20),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
                     

                   
if __name__ == '__main__':
    
    start = time.time()
    
    svm1 = cv2.ml.SVM_load(svm1)
    svm_num = cv2.ml.SVM_load(svm2)
    
    category = categories[5]

    directory = '/home/rima/Dokumente/BA/dataset/validate'
    
    path = os.path.join(directory,category)  

    for image in os.listdir(path):
                  
        print("processing image " + image)
        
        bgr = cv2.imread(os.path.join(path, image))           
        rgb = cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB) 

        detect(rgb)
        
    

    #validation
    correctDetected, detectedSigns, trueSigns, predictListSigns,signs = return_validation()    
    precision = (correctDetected / detectedSigns) * 100
    recall = (correctDetected / trueSigns) * 100
    print(correctDetected, trueSigns, detectedSigns)
    print(precision, recall) 

    confusion_mat()    
        
        
    print("%s secondes" % (time.time() - start))                            
