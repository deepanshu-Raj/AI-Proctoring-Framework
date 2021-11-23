import os
import cv2
import pickle
import argparse
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from imutils import paths
from model_script.liveness_model import LivenessNet
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Added argument parser to get more detailed info from command line while program run

# --dataset : path to input dataset.
# --model : path to trained model.
# --le : path to label encoder.
# --plot : path to output loss/accuracy plot.

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,help="path to input dataset")
ap.add_argument("-m", "--model", type=str, required=True,help="path to trained model")
ap.add_argument("-l", "--le", type=str, required=True,help="path to label encoder")
ap.add_argument("-p", "--plot", type=str, default="TrainingLoss&AccuracyOnDatasetPlot.png",help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

# initialize the initial learning rate, batch size, and number of epochs to train for
INIT_LR = 1e-4
BS = 8
EPOCHS = 50

# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

# loop over all image paths
for imagePath in imagePaths:
	# Extract the class label from the filename.
	label = imagePath.split(os.path.sep)[-2]

	# Loading the image + Resizing it to 32x32.
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (32, 32))

	# Update the data and labels lists, respectively
	data.append(image)
	labels.append(label)

# normalizing the pixel values to get in range - [0.0,1.0]
data = np.array(data, dtype="float") / 255.0

# encode the labels and one-hot encode them
le = LabelEncoder()
labels = le.fit_transform(labels)
labels = to_categorical(labels, 2)

# partitioning the data into train - test split : ratio 0.75 (train), 0.25 (test)
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)


# construct the training image generator for data augmentation
# aug is the image augmentation object, helps us in adding dummy data. 
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15, width_shift_range=0.2, 
						 height_shift_range=0.2, shear_range=0.15, horizontal_flip=True, 
						 fill_mode="nearest")


# Initialize the optimizer and model - Adam optimizer & LivenessNet Model.
print("[INFO] compiling model...")
opt = Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
model = LivenessNet.build(width=32, height=32, depth=3,classes=len(le.classes_))
model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])


# Model Training Step
print("[INFO] training network for {} epochs...".format(EPOCHS))
H = model.fit(x=aug.flow(trainX, trainY, batch_size=BS),validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS, epochs=EPOCHS)


# Model Evaluation
print("[INFO] evaluating network...")
predictions = model.predict(x=testX, batch_size=BS)
print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1), target_names=le.classes_))


# SAVING MODEL & LABEL ENCODER

# save the Model to disk
print("[INFO] serializing network to '{}'...".format(args["model"]))
model.save(args["model"], save_format="h5")


# save the label encoder to disk
with open(args["le"], "wb") as f:
	f.write(pickle.dumps(le))
f.close()



# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, EPOCHS), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, EPOCHS), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, EPOCHS), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, EPOCHS), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])