#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
train svm to classify six traffic signs
"""

from sklearn.metrics import classification_report,accuracy_score
import numpy as np
import cv2
import os
import image_process
import matplotlib.pyplot as plt

features = []
labels = []

dic_shapes = { "nosign": 0, "30": 1, "50":2, "60":3, "stop": 4, "priority": 5, "yield": 6}

def train_svm1(directory):
    
    global categories, features, labels, dic_shapes
    
    
    for category in dic_shapes:
       path = os.path.join(directory,category)
       
       for image in os.listdir(path):
            roiListR = []
            roiListY = []
            bgr = cv2.imread(os.path.join(path, image))
            # resize images of dataset
            bgr = cv2.resize(bgr,(400,300))
            
            rgb = cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
#            plt.imshow(rgb)
#            plt.show() 
            
            # segmentation
            masks = image_process.color_segmentation(rgb)
            
            if category != "priority":
                statsListR,roiListR = image_process.get_roi(masks[0],rgb, 1)
                
            if category == "priority" or category == "nosign" : 
                statsListY,roiListY = image_process.get_roi(masks[1],rgb, 2)
            
            print(image)
            
#           for training use only the first roi in list
            if len(roiListR) > 0:
                feature = image_process.hogFeature(roiListR[0],(60,60), 8,(5,5), (2,2), 'L2')
                if feature is not None:  

                    features.append(feature)
                    num = dic_shapes[category]
                    
                    labels.append(num)

#            
            if len(roiListY) > 0:
                 feature = image_process.hogFeature(roiListY[0],(60,60), 8,(5,5), (2,2), 'L2')
                 if feature is not None: 

                    features.append(feature)
                    num = dic_shapes[category]
                    labels.append(num)


    features = np.float32(features)           
    labels = np.array(labels)

    train()  
        
        

def train():
    
    global features, labels

    dirpath = os.getcwd()
    pathmodel = "/svm1.xml"
    filename = dirpath+pathmodel


    rand = np.random.RandomState(21)
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

    train_svm1(directory)
