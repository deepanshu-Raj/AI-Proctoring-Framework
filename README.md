# AI-Proctoring Framework

As the mode of education has completely shifted from class-based learning to Online Education, levels of bias in defrauders have increased while attempting examinations. Manual or human proctoring is neither feasible nor an appropriate technique to proctor candidates in online examinations.


## Idea

Idea is to build a tool that runs in the background on the examinee’s machine, and tracks any kind of unwanted activity of the candidate. The tool will be preset to give 2 warnings to the candidate, to which if he/she disobeys, will bring the candidate’s test to end. As the name suggests, It enables the possiblity of conducting examination,with proctors being Machines themselves.
## Working

- As soon as the test starts at the student's end, the system's camera at the candidate's end starts capturing the video frames. 

- Upon this video frame, the frontal_face_detector detects the count of faces in the frame; blink detection script, evaluates if the candidate is blinking & the Iris movements of the candidate is tracked , using gaze detection and face_detection_68_facial_landmarks data model.

- Object Detection is also enabled in the test envoirnment.

- Via a fine-tuned function, this script considering all the factors(blink, gaze, object detection, facial counts..), calculates a suspicion value, which is then compared with a Threshold value, and if the suspicion value exceeds the threshold value; the student can be given a warning, the 3rd time it happens, test can be ended automatically.

## Features

- ✅ Mouth Tracking
- ✅ Eye Tracking
- ✅ Gaze Detection
- ✅ Liveliness Detection
- ✅ Object Detection
- ❌ Voice Proctoring

## Authors

- [@Deepanshu Raj](https://github.com/deepanshu-Raj)
- [@Devansh Shukla](https://github.com/devanshjsr)
- [@DharmRaj Maurya](https://github.com/Dharm1999)

## Demonstration

<p align="center">

<img src="https://github.com/deepanshu-Raj/AI-Proctoring-Framework/blob/master/1-1.gif" height=425 width=600 alt="Face & Blink Detection">
<h3 align="center">Canvas 1 : Face & Blink Detection</h3>

</p>
<br>
<br>
<p align="center">
 
<img src="https://github.com/deepanshu-Raj/AI-Proctoring-Framework/blob/master/1-2.gif" height=425 width=600 alt="Gaze & Liveness Detection">
<h3 align="center">Canvas 2 : Gaze & Liveness Detection</h3>
 
</p>

