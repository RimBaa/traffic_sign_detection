#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
create ROS node for traffic sign detection

get image from the camera of a model car
detect traffic signs
publish image witch detected traffic signs to topic traffic_sign
"""
import roslib
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import time
import os
import cv2
import image_process
import numpy as np

categories = ["nosign", "30", "50", "60", "stop", "priority", "yield" ]
dic_shapes = { "nosign": 0, "30": 1, "50": 2, "60": 3, "stop": 4, "priority": 5, "yield": 6}


class detect_traffic_sign:
# path to the trained svm models
    dirpath = os.getcwd()
    pathmodel1 = "/src/detect_traffic_sign/model/svm1.xml"
    pathmodel2 = "/src/detect_traffic_sign/model/svm_number.xml"
    svm1_path = dirpath+pathmodel1
    svm2_path = dirpath+pathmodel2
    
    svm_shape = cv2.ml.SVM_load(svm1_path)
    svm_num = cv2.ml.SVM_load(svm2_path)
        
        
    def __init__(self):
        self.image_pub = rospy.Publisher("/traffic_sign", Image, queue_size = 1)
        
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/sensors/camera/color/image_raw", Image, self.callback, queue_size = 1, buff_size = 2**24)
        
    def callback(self, data):

        start = time.time()
        try:
            rgb = self.bridge.imgmsg_to_cv2(data, "rgb8")
        except CvBridgeError as e:
            print(e)
        
        im_signs = self.detect(rgb)
        
        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(im_signs, "rgb8"))
            print("%s secondes" % (time.time() - start))

        except CvBridgeError as e:
            print(e)
            
            
    def detect(self,rgb):

       
        print("processing image ")
        features = []
        index = 0
            
        # resize images of dataset
        rgb = cv2.resize(rgb,(400,300))
        gray = cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    
        #red, yellow segmentation + ROIs selection
        masks = image_process.color_segmentation(rgb)
                
        statsListR,roiListR = image_process.get_roi(masks[0],rgb, 1)
        
        statsListY,roiListY = image_process.get_roi(masks[1],rgb, 2)
        statsList = statsListR + statsListY
        roiList = roiListR + roiListY
        
      # HOG feature of shape
        if len(roiList) > 0:
            for idx, roi in enumerate(roiList):

                feature = image_process.hogFeature(roi,(60,60), 9,(8,8), (2,2), 'L2')
    
                if feature is not None:  
                    features.append(feature)
            
        if len(features) > 0:    
    # predict sign
            features = np.float32(features)
    
            predicted = self.svm_shape.predict(features)[1].ravel()
    
            #speed limit -> predict number
            for idx, predict in enumerate(predicted):
                if predict == 1 or predict ==2 or predict ==3:
                    feat= []
                    img = np.uint8(gray[statsList[idx][1]:statsList[idx][3], statsList[idx][0]:statsList[idx][2]])
                    img = cv2.equalizeHist(img)

                    numIm = image_process.hogFeature(img, (100,100),9,(6,6),(2,2),'L2')

                    feat.append(numIm)
                    predictNum = self.svm_num.predict(np.float32(feat))[1].ravel()
                    predicted[idx]= predictNum
    
    # bounding box around detected signs
            for index,predict in enumerate(predicted):
                if predict > 0:
                    self.draw_boundingBox(statsList[index], predicted[index],rgb)
          
            
            print("predicted signs: %s" % predicted)
          
        return rgb                             

    def draw_boundingBox(self,roi, trafficSign, rgb):
        
        sign = int(trafficSign)
        
        cv2.rectangle(rgb,(roi[0],roi[1]),(roi[2],roi[3]),(0,0,255),2)
        cv2.putText(rgb,categories[sign],(roi[0],roi[3]+20),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
                
    
        
def main():
   
   rospy.init_node('traffic_sign_detector', anonymous = True)
   
   im = detect_traffic_sign()
   try:
       rospy.spin()
   except KeyboardInterrupt:
       print("Shutting down")
   
if __name__ == '__main__':
    main()
                     
