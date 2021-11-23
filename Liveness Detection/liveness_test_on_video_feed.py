import os
import cv2
import time
import pickle
import imutils
import argparse
import numpy as np
from imutils.video import VideoStream
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model


# Added argument parser to get more detailed info from command line while program run

# --model : path to trained liveness model.
# --le : path to label encoder.
# --detector : path to openCV's deep learning face detector.
# --confidence : threshold probability for weak detections filtering.

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=True, help="path to trained model")
ap.add_argument("-l", "--le", type=str, required=True, help="path to label encoder")
ap.add_argument("-d", "--detector", type=str, required=True, help="path to OpenCV's deep learning face detector")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
args = vars(ap.parse_args())


# Loading openCV's deep learning face detector.
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"], "res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)


print("[INFO] loading liveness detector...")
# loading the trained liveness detector model.
model = load_model(args["model"])
# loading the label encoder.
le = pickle.loads(open(args["le"], "rb").read())


# starting the video stream
print("[INFO] starting video stream...")
video_stream = VideoStream(src=0).start()
time.sleep(2.0)


# Start working on video feed.
while True:
	
	# Reading frame & resizing it's width to 600 px.
	frame = video_stream.read()
	frame = imutils.resize(frame, width=600)

	# grab the frame dimensions and convert it to a blob of 300x300 dimensions.
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
	
	# passing the image blob to the openCV's deep nn to get the detections. 
	net.setInput(blob)
	detections = net.forward()

	# loop through all the detections obtained.
	for i in range(0, detections.shape[2]):
		# confidence is the detection's accuracy measure. 
		confidence = detections[0, 0, i, 2]
		# filter out weak detections
		if confidence > args["confidence"]:
			# compute the (x, y)-coordinates of the bounding box for the face and extract the face ROI
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# The bounding box must always lie inside the Video frame.
			startX = max(0, startX)
			startY = max(0, startY)
			endX = min(w, endX)
			endY = min(h, endY)
			
			# Face ROI is extracted from the frame and is resized into 32x32 image.
			face = frame[startY:endY, startX:endX]
			face = cv2.resize(face, (32, 32))
			# Pixels of this 32x32 image is normalised [0.0,1.0] to get the final face image.
			face = face.astype("float") / 255.0
			
			# processing the face (ROI).
			face = img_to_array(face)
			face = np.expand_dims(face, axis=0)
			
			# face ROI is passed through the trained liveness detector model to determine if the face is "real" or "fake"
			preds = model.predict(face)[0]
			pred_label = np.argmax(preds)
			label = le.classes_[pred_label]

			# draw the label and bounding box on the frame
			label_text = "{}: {:.4f}".format(label, preds[pred_label])
			cv2.putText(frame, label_text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
			cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

	cv2.imshow("Frame", frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# Destroying all windows & stop the video stream.
cv2.destroyAllWindows()
video_stream.stop()