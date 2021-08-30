import cv2
from math import hypot
import numpy as np
import dlib

predictor = dlib.shape_predictor("../Landmarks/shape_predictor_68_face_landmarks.dat")

def distance(point_A,point_B):
	"""
	Input :
    Two localised points from the candidate's Face
	    point_A : tuple of co-ordinates (x,y)
	    point_B : tuple of co-ordinates (x,y)
    
    Output :
    Returns the distance between these two points
    
    Action :
    Finds the Eucledian distance between point_A & point_B
	"""
	dist = hypot((point_A[0]-point_B[0]),(point_A[1]-point_B[1]))
	return dist

def mouthTrack(faces,frame):
	"""
	Input :
	    faces : list of frontal_face_detector() objects which corresponds to the face objects detected in the video frame.
	    frame : Video Frame

    Output :
    <!-- void -->
    
    Action :
    will detect if the candidate's mouth is Open or not, & display the same on the video Frame.
	"""
	global predictor

	for face in faces:

		landmarks = predictor(frame,face)

		# outer_top_x : outer lip top point's x-coord
		# outer_top_y : outer lip top point's y-coord
		outer_top_x = landmarks.part(51).x
		outer_top_y = landmarks.part(51).y

		# outer_bottom_x : outer lip bottom point's x-coord
		# outer_bottom_y : outer lip bottom point's y-coord
		outer_bottom_x = landmarks.part(57).x
		outer_bottom_y = landmarks.part(57).y

		dist = distance((outer_top_x,outer_top_y),(outer_bottom_x,outer_bottom_y))
		if(dist>30):
			cv2.putText(frame,"MOUTH OPEN - "+str(dist),(50,80),cv2.FONT_HERSHEY_PLAIN,2,(0,255,255),5)

		cv2.putText(frame,"THRESHOLD - "+str(30),(50,400),cv2.FONT_HERSHEY_PLAIN,2,(0,255,255),5)