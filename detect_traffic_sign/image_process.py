#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
image preprocessing:

color segmentation
ROI extraction
HOG feature of ROIs

"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage.feature import hog


#red, yellow color segmentation
def color_segmentation(rgb):

    hsvIm =cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)

    h, s, v = cv2.split(hsvIm)

    if np.mean(v) >= 100:
        brightness = 100
    else:
        brightness = 70
    hsvIm = cv2.merge((h, s, v))

#  red color segmentation
    low_red = np.array([0,100,brightness])
    high_red = np.array([8,255,255])
    
    maskRed1 = cv2.inRange(hsvIm,low_red,high_red)

    lower_red = np.array([158,75,brightness])
    upper_red = np.array([179,255,255])
    
    maskRed2 = cv2.inRange(hsvIm, lower_red, upper_red)

    maskRed = maskRed1+maskRed2
    
#   yellow color segmentation     
    low_yellow = np.array([8,90,90])
    high_yellow = np.array([38,235,250])
    maskYellow = cv2.inRange(hsvIm,low_yellow,high_yellow)
    
#    show segmentation results
#    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
#    
#    ax1.axis('off')
#    ax1.imshow(maskRed)
#    ax1.set_title('Red Segmentation')
#    
#    ax2.axis('off')
#    ax2.imshow(maskYellow)
#    ax2.set_title('Yellow Segmentation')
#    plt.show()
    
    
    return (maskRed,maskYellow)


# filter nd extract ROIs
def get_roi(image, rgb, color):

    
    gray = cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    
    _, _, stats, _ = cv2.connectedComponentsWithStats(image, connectivity =8 )

#   remove background
    stats = np.delete(stats,0,0)

    sizes = stats[:,-1]
    roiList = []
    statsList = []

#define min size for red and yellow rois
    if color == 1:
        min_size =25
        min_length = 20

    elif color == 2:
        min_size =10
        min_length = 9
        
#   remove ROIs that are too small
    sizes = (np.where(sizes >= min_size))[0]    
        
    for roi in sizes: 

        if (stats[roi] >= 0).all() == True :
  
                height = stats[roi,cv2.CC_STAT_HEIGHT]
                width = stats[roi, cv2.CC_STAT_WIDTH]
                proportion = max(height,width)*1.0/min(height,width)

                # checking if height and width have nearly the same length
                if  ( proportion< 1.4) and height > min_length and width > min_length:

                    top = stats[roi,cv2.CC_STAT_TOP]   
                    left = stats[roi,cv2.CC_STAT_LEFT]
                    
                    # adding a small border to make sure the whole sign is in the ROI
                    if color == 1: 
                        scale_percentage = 10   
                        
                    #priority road sign has a white border that needs to be included in the ROI        
                    elif color == 2:
                        scale_percentage = 50
                        
                    addwidth = stats[roi,cv2.CC_STAT_WIDTH] *scale_percentage // 100
                    addheight = stats[roi, cv2.CC_STAT_HEIGHT] * scale_percentage // 100
                        
                    if top - addheight >= 0:
                        top -= addheight             
                    if left -addwidth >=0:
                        left -= addwidth
                    if height+ addheight <= image.shape[0]:
                        height += addheight
                    if width+ addwidth <= image.shape[1]:
                        width += addwidth 
                    
                    #extract ROIs      
                    roiIm = np.uint8(image[top:stats[roi,cv2.CC_STAT_TOP]+height, left:stats[roi,cv2.CC_STAT_LEFT]+width])
                    grayroi = np.uint8(gray[top:stats[roi,cv2.CC_STAT_TOP]+height, left:stats[roi,cv2.CC_STAT_LEFT]+width])   

                    #remove background
                    if color == 1:
                        h,w = roiIm.shape[:2]
                        mask = np.zeros((h+2,w+2), np.uint8)
                        cond =roiIm>100
                        
                        cv2.floodFill(roiIm,mask, (0+addheight,0+addwidth), 100)

                        pixels =  np.where(cond, grayroi, roiIm)
                        pixels = np.where(pixels<100, grayroi, pixels)
                        roiList.append(pixels)
                    else:
                        roiList.append(roiIm)
                        
                    statsList.append([left, top ,stats[roi,cv2.CC_STAT_LEFT] + width, stats[roi,cv2.CC_STAT_TOP] + height])

    return statsList,roiList

# compute histogram of oriented gradients
def hogFeature(roi, size, orientation, pixels, cells, norm):

    gray = cv2.resize(roi,size)
    fd, hog_image = hog(gray, orientations=orientation, pixels_per_cell=pixels,
                        cells_per_block=cells, block_norm= norm, visualize=True, transform_sqrt=False, feature_vector= True,  multichannel=False)
    
#    show image and HOG descriptor
#    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
#    
#    ax1.axis('off')
#    ax1.imshow(gray, cmap=plt.cm.gray)
#    ax1.set_title('Input')
    
#    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
#    
#    ax2.axis('off')
#    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
#    ax2.set_title('Histogram of Oriented Gradients')
#    plt.show()


    return fd
