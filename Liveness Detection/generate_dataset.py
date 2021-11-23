# import the necessary packages
import os
import cv2
import numpy as np
import argparse


# Added argument parser to get more detailed info from command line while program run
# --input : path to input video file
# --output : path where images needs to be stored.
# --detector : Face detector model
# --confidence : Only accept face with confidence score > -c tag value
# --skip : no. of frames to skip.

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, required=True,
	help="path to input video")
ap.add_argument("-o", "--output", type=str, required=True,
	help="path to output directory of cropped faces")
ap.add_argument("-d", "--detector", type=str, required=True,
	help="path to OpenCV's deep learning face detector")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-s", "--skip", type=int, default=16,
	help="# of frames to skip before applying face detection")
args = vars(ap.parse_args())


# load our serialized face detector from disk
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],"res10_300x300_ssd_iter_140000.caffemodel"])

print('model loaded successfully!!')
# Out Deep neural net to detect faces from the video stream.
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# open a pointer to the video file stream 
# Total number of frames read and saved so far (initialize = 0)
video_stream = cv2.VideoCapture(args["input"])
frames_read = 0
frames_saved = 0

# loop over frames from the video file stream
# This loop breaks when all the frames in the video stream is read
while True:

	# grab the frame from the file
	(grabbed, frame) = video_stream.read()

	# if the frame was not grabbed, then we have reached the end of the stream
	if not grabbed:
		break
	
	# increment the total number of frames read thus far
	frames_read += 1

	# check to see if we should process this frame
	if frames_read % args["skip"] != 0:
		continue

	# Frame dimensions
	(h, w) = frame.shape[:2]
	# Creating the blob from the frame - which will be fed to the Deep neural network.
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))
	# blob is passed through the network and the detections and predictions are obtained.
	net.setInput(blob)
	detections = net.forward()


	# ensure at least one face was found
	if len(detections) > 0:
		# we're making the assumption that each image has only ONE
		# face, so find the bounding box with the largest probability
		i = np.argmax(detections[0, 0, :, 2])
		confidence = detections[0, 0, i, 2]

		# Only add the image into dataset if the confidence score is > than the confidence score provided.
		if confidence > args["confidence"]:
			# compute the (x, y)-coordinates of the bounding box for the face and extract the face ROI
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			
			# Extracted Face from the video stream.
			face = frame[startY:endY, startX:endX]
			# write the frame to disk - to the dataset folder
			file_path = os.path.sep.join([args["output"],"{}.png".format(frames_saved)])
			cv2.imwrite(file_path, face)
			frames_saved += 1
			print("[INFO] saved {} to disk".format(file_path))

# Releasing the video stream & Destroying all the windows.
video_stream.release()
cv2.destroyAllWindows()