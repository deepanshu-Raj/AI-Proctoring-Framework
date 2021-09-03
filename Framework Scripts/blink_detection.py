import cv2
import numpy as np
import dlib
from math import hypot
from facial_landmarks_detection import *

def midPoint(pointA,pointB):
    """
    Input : two points A,B.
    Output : Midpoint of these two points
    """
    
    X = int(pointA.x+pointB.x)/2
    Y = int(pointA.y+pointB.y)/2
    
    return (X,Y)

def findDist(pointA,pointB):
    """
    Input : two points A, B.
    Output : Eucledian Norm of these two points.
    """
    
    dist = hypot((pointA[0]-pointB[0]),(pointA[1]-pointB[1]))
    return dist

def isBlinking(faces,frame):
    """
    Input : frame from the video stream, and faces : from the localisation if faces in the screen.
    Output : tuple ~ (n1,n2,str)
            n1 : left eye Ratio(horLen/verLen)
            n2 : right eye Ratio(horLen/verLen)
            str : 'Blink' or 'no Blink'
            
    <!-- Uncomment the commented section to find more functionality(specific to a particular eye) -->
    """
    
    font = cv2.FONT_HERSHEY_PLAIN
    ratio = ()
    thickness = 5
    
    
    #these points are written w.r.t the camera capture. [left<->right]
    right = [36,37,38,39,40,41]
    left = [42,43,44,45,46,47]
    
    for face in faces:
        
        facialLandmarks = shapePredictor(frame,face)
        
        #right eye markings        
        rLeftPoint = (facialLandmarks.part(36).x,facialLandmarks.part(36).y)
        rRightPoint = (facialLandmarks.part(39).x,facialLandmarks.part(39).y)
        rTopPoint = midPoint(facialLandmarks.part(37),facialLandmarks.part(38))
        rBottomPoint = midPoint(facialLandmarks.part(40),facialLandmarks.part(41))
        
        rightHorLen = findDist(rLeftPoint,rRightPoint)
        rightVerLen = findDist(rTopPoint,rBottomPoint)
        
        #left eye markings
        lLeftPoint = (facialLandmarks.part(42).x,facialLandmarks.part(42).y)
        lRightPoint = (facialLandmarks.part(45).x,facialLandmarks.part(45).y)
        lTopPoint = midPoint(facialLandmarks.part(43),facialLandmarks.part(44))
        lBottomPoint = midPoint(facialLandmarks.part(46),facialLandmarks.part(47))
        
        leftHorLen = findDist(lLeftPoint,lRightPoint)
        leftVerLen = findDist(lTopPoint,lBottomPoint)
        
        # Calculating the ratios of left and right eye's vertical and horizontal lengths.
        lRatio = leftHorLen/leftVerLen
        rRatio = rightHorLen/rightVerLen
        
        # optimal threshold for a blink comes to be around 5.1
        # hence, if the ratio >= 5.1 : blink else noBlink.
        
        if(lRatio>=5.1 or rRatio>=5.1) :
            cv2.putText(frame,"blink",(50,140),font,4,(0,255,255),thickness)
            ratio += (lRatio,rRatio,'Blink')
        else:
            ratio += (lRatio,rRatio,'No Blink')
        
        # for indivisual eye
        # if(lRatio>=5.1) :
        #     cv2.putText(frame,"LEFT",(50,90),font,7,(0,255,255),thickness)
        # if(rRatio>=5.1) :
        #     cv2.putText(frame,"RIGHT",(50,90),font,7,(0,255,255),thickness)
    
    return ratio