import cv2
import numpy as np
import dlib

shapePredictorModel = "../Landmarks/shape_predictor_68_face_landmarks.dat"
shapePredictor = dlib.shape_predictor(shapePredictorModel)

def detectFace(frame):
    """
    Input :
    will receive a video frame, from the frontal camera
    
    Output :
    returns the count of faces detected by the dlib's
    default frontal face detector.
    
    Action :
    will detect all the faces and localize them in a rectangular box.
    """
    faceDetector = dlib.get_frontal_face_detector()
    faces = faceDetector(frame)
    
    # faces now contains the coords of the rectangles,
    # by which the faces are bound.
    # [(x1,y1) (x2,y2)] where pt.1 ~ top left coords
    # pt.2 ~ bottom right coords
    
    faceCount = len(faces)
    # uncomment till cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,255),3) to view the localised face in the frame.
    
    # for face in faces:
    #     x1 = face.left()
    #     y1 = face.top()
        
    #     x2 = face.right()
    #     y2 = face.bottom()
        
    #     faceCount += 1
        
        # cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,255),3)
        
    return (faceCount,faces)


def landmarkLocalisation(faces,frame):
    """
    Input : 
    faces : Faces localized in a rectangle from detectFace() function.
    frame : Video frame obtained.
    
    Output :
    <!-- void -->
    
    Action :
    will mark all the 68 facial landmarks on the face detected.
    """
    
    for face in faces:
        
        # giving our detected face and the video frame, to our shape predictor.
        # this will create a 68 part object, with each part containing (x,y) coord, of the indivisual facial landmark.
        facialLandmarks = shapePredictor(frame,face)
        
        # looping through all the 68 points of the facial landmarks
        for i in range(68):
            x_coord = facialLandmarks.part(i).x
            y_coord = facialLandmarks.part(i).y
            
            # plotting each of the faial landmark on the video frame,as a circle with rad = 3
            cv2.circle(frame,(x_coord,y_coord),3,(255,0,0),-1)

