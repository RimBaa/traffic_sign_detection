#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
train svm to classify speed limit signs
"""


from sklearn.metrics import classification_report,accuracy_score
import numpy as np
import cv2
import os
import image_process
import matplotlib.pyplot as plt

features = []
labels = []

dic_speed = {"nosign": 0, "30": 1, "50": 2, "60":3}

def train_svm_num(directory):
    
    global features, labels, dic_speed
    
    for category in dic_speed:

       path = os.path.join(directory,category)
       
       for image in os.listdir(path):
            roiListR = []

            bgr = cv2.imread(os.path.join(path, image))
            
            # resize image
            bgr = cv2.resize(bgr,(400,300))
            
            rgb = cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
            
            # segmentation
            masks = image_process.color_segmentation(rgb)
            

            statsListR,roiListR = image_process.get_roi(masks[0],rgb, 1)

            print(image)

            
            gray = cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
#                for training use only the first roi in list
            if len(roiListR) > 0:
                
                 img = np.uint8(gray[statsListR[0][1]:statsListR[0][3], statsListR[0][0]:statsListR[0][2]])
                 img = cv2.equalizeHist(img)                
                
                 feature = image_process.hogFeature(img, (100,100),9,(6,6),(2,2),'L2')
                 
                 if feature is not None:  
                    features.append(feature)
                    num = dic_speed[category]
                    labels.append(num)

    features = np.float32(features)
        
    labels = np.array(labels)

    train()  
        
#                  
    
        

def train():
    
    global features, labels
    dirpath = os.getcwd()
    pathmodel = "/svm_number.xml"
    filename = dirpath+pathmodel
  

    rand = np.random.RandomState(321)
    shuffle = rand.permutation(len(features))

    features = features[shuffle]
    
    labels = labels[shuffle]
    
    percentage = 90
    partition = int(len(features)*percentage/100)

    x_train, y_train = features[:partition], labels[:partition]
    x_test, y_test = features[partition:], labels[partition:]

    svm = cv2.ml.SVM_create()

    svm.setType(cv2.ml.SVM_C_SVC)

    svm.setKernel(cv2.ml.SVM_INTER)

    svm.setC(10)

    svm.setGamma(0.01)
     
    svm.train(x_train, cv2.ml.ROW_SAMPLE, y_train)
     
    svm.save(filename);
     
    testResponse = svm.predict(x_test)[1].ravel()
    
    print(testResponse)
    print(y_test)
    
    print("Accuracy: "+str(accuracy_score(y_test, testResponse)))
    print('\n')
    print(classification_report(y_test, testResponse))

  
if __name__ == '__main__':
    dirpath = os.getcwd()
    path_data = "/train_data"
    directory = dirpath+path_data

    train_svm_num(directory)

