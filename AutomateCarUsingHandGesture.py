# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 03:57:31 2020

@author: Ankit
"""

import cv2
import numpy as np
import math
import pyautogui as pyg

#direction variables
l = 0 #left
r = 0 #right
u = 0 #up
d = 0 #down


learningRate = 0
blurValue = 41
threshold = 60 
cap = cv2.VideoCapture(0)
isBgCaptured = 0   # bool variable , whether the background is captured ot not
bgSubThreshold = 50

def removeBG(crop_image):
    fgmask = bgModel.apply(crop_image,learningRate=learningRate)
    
    kernel = np.ones((3, 3), np.uint8)
    
    """-->morphological transformations like erode are normally done on binary images.
        -->In erode,kernel slides through every pixel,if all pixels in kernel are 1,
        then that pixel turns to 1 otherwise turns to 0(tries to keep forground white)
        -->used for removing white noises,detaching two connected forground objects.
    """
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(crop_image, crop_image, mask=fgmask)
    cv2.imshow('ttttt',res)
        
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
    cv2.imshow('blur', blur)
    ret, thresh1 = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)
    cv2.imshow('ori', thresh1)
    return thresh1

while cap.isOpened():
    
    ret,frame = cap.read()
    cv2.rectangle(frame, (70, 70), (320, 320), (0, 255, 255), 3)
    crop_image = frame[70:320, 70:320]
    cv2.imshow('frame',frame)
    if isBgCaptured == 1:
        
        thresh = removeBG(crop_image)
        
        """
        -->Contours are curve/s joining all the 
        continuous points (along the boundary), having same color or 
        intensity.
        -->To find contours,we use binary images for better accuracy(threshold)
        """
    
        """
        Contours is a Python list of all the contours in the image. 
        Each individual contour is a Numpy array of 
        (x,y) coordinates of boundary points of the object.
        """
        
        image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
        try:
            # Find contour with maximum area
            contour = max(contours, key=lambda x: cv2.contourArea(x))
    
            # Create bounding rectangle around the contour
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(crop_image, (x, y), (x + w, y + h), (0, 0, 255), 1)
    
            # Find convex hull
            hull = cv2.convexHull(contour)
    
            # Draw contour
            drawing = np.zeros(crop_image.shape, np.uint8)
            #draws the contours inside the hull
            cv2.drawContours(drawing, [contour], -1, (0, 255, 0), 2)
            #draws the exterior hull
            cv2.drawContours(drawing, [hull], -1, (0, 0, 255), 3)
    
            # # Find convexity defects (we use returnPoints=False to find convexity defects)
            hull = cv2.convexHull(contour, returnPoints=False)
            
            """
            convexityDefects--> returns an array where each row 
            contains these values - 
            [ start point, end point, farthest point, 
             approximate distance to farthest point ].
            """
            defects = cv2.convexityDefects(contour, hull)
    
            # Use cosine rule to find angle of the far point from the start and end point i.e. the convex points (the finger
            # tips) for all defects
            count_defects = 0
    
            #s,e,f --> indices of contour
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(contour[s][0])
                end = tuple(contour[e][0])
                far = tuple(contour[f][0])
    
                a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                angle = (math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 180) / 3.14
    
                # if angle > 90 draw a circle at the far point
                if angle <= 90:
                    count_defects += 1
                    cv2.circle(crop_image, far, 8, [211, 84, 0], -1)
    
                cv2.line(crop_image, start, end, [0, 255, 0], 2)
    
            # Print number of fingers
            if count_defects == 0:#release all Buttons / Works as slowing down or brake
                cv2.putText(frame, "ONE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255),2)#one finger detected
                if l == 1:#release left
                    pyg.keyUp('left')
                    l = 0
                if r == 1:#release right
                    pyg.keyUp('right')
                    r = 0
                if u == 1:#release up
                    pyg.keyUp('up')
                    u = 0
                if d == 1:#release down
                    pyg.keyUp('down')
                    d = 0
            elif count_defects == 1:#Accelerate
                cv2.putText(frame, "TWO", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255), 2)#two fingers detected
                if l == 1:#release left
                    pyg.keyUp('left')
                    l = 0
                if r == 1:#release right
                    pyg.keyUp('right')
                    r = 0
                if u == 0:#hold up
                    pyg.keyDown('up')
                    u = 1
                if d == 1:#release down
                    pyg.keyUp('down')
                    d = 0
            elif count_defects == 2:#Go Right
                cv2.putText(frame, "THREE", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255), 2) #three fingers detected
                if l == 0:#hold left
                    pyg.keyDown('left')
                    l = 1
                if r == 1:#release right
                    pyg.keyUp('right')
                    r = 0
                if u == 1:#release up
                    pyg.keyUp('up')
                    u = 0
                if d == 1:#hold down
                    pyg.keyUp('down')
                    d = 0
            elif count_defects == 3:#Go Left
                cv2.putText(frame, "FOUR", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255), 2) #4 fingers detected
                if l == 1:#release left
                    pyg.keyUp('left')
                    l = 0
                if r == 0:#hold right
                    pyg.keyDown('right')
                    r = 1
                if u == 1:#release up
                    pyg.keyUp('up')
                    u = 0
                if d == 1:#release down
                    pyg.keyUp('down')
                    d = 0
                
            elif count_defects == 4:#hit nitro boost
                cv2.putText(frame, "FIVE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255), 2) #5 fingers detected
                pyg.press('x')
            else:
                pass
        except:
            pass
    
        # Show required images
        cv2.imshow("Gesture", frame)
        all_image = np.hstack((drawing, crop_image))
        cv2.imshow('Contours', all_image)
        
        
    
    k = cv2.waitKey(10)
    if k == 27:  # press ESC to exit
        cap.release()
        cv2.destroyAllWindows()
        break
    elif k == ord('b'):  # press 'b' to capture the background
        bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
        isBgCaptured = 1
        print( '!!!Background Captured!!!')
    
cap.release()
cv2.destroyAllWindows()
    