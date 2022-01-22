<h1 align="center">AI-Proctoring-Framework</h1>
<h3 align="center">B.Tech Final Year Project - I</h3>
<p align="center"><code><b>Do "⭐" the repository, if you liked the work</b></code></p>
<br><br>

<img align="right" src="https://user-images.githubusercontent.com/54600788/128983568-928da7b1-a365-49d9-be17-01a94add0fcb.png"></img>

### <img height="22" width="22" src="https://user-images.githubusercontent.com/54600788/128984688-6b040caf-a5ce-4a00-94ae-7d52c21535c5.png"></img> &nbsp; Project Group : 

  * <code>[Deepanshu Raj](https://github.com/deepanshu-Raj)</code>
  * <code>[Devansh Shukla](https://github.com/devanshjsr)</code>
  * <code>[DharmRaj Maurya](https://github.com/Dharm1999)</code>

<br>

### Background :

<p>As the mode of education has completely shifted from class-based learning to Online Education, levels of bias in defrauders have increased while attempting examinations. Manual or human proctoring is neither feasible nor an appropriate technique to proctor candidates in online examinations.</p>

<br>

<img height = "200" align="left" src="https://user-images.githubusercontent.com/54600788/128985687-f1d248d9-1c39-4426-b88e-bdb7f46cb7bf.png"></img>


### Idea :
Idea is to build a tool that runs in the background on the examinee’s machine, and tracks any kind of unwanted activity of the candidate. The tool will be preset to give 2 warnings to the candidate, to which if he/she disobeys, will bring the candidate’s test to end. As the name suggests, It enables the possiblity of conducting examination,with proctors being Machines themselves.</p>

<br>

### Working :
<img height = "200" align="right" src="https://user-images.githubusercontent.com/54600788/128986198-c45a1db1-0c62-42d3-b47a-7a43827b2066.png"></img>

<p>
It works as in, <b>As soon as the test starts at the student's end, the system's camera at the candidate's end starts capturing the video frames</b>. Upon this video frame, the <b>frontal_face_detector detects the count of faces in the frame</b>; blink detection script, evaluates if the <b>candidate is blinking (or ultimately looking down)</b> & the <b>Iris movements of the candidate is tracked , using gaze detection and face_detection_68_facial_landmarks data model</b>. Object Detection is also enabled in the test envoirnment.
</p>

<p>
  Via a <b>fine-tuned function</b>, this script considering <b>all the factors(blink, gaze, object detection, facial counts..)</b>, calculates a suspicion value, which is then compared with a Threshold value, and <b>if the suspicion value exceeds the threshold value; the student is given a warning( 2 times ), the 3rd time it happens,<br> test ends automatically.</b>
</p>
<br>

### 📌 AI-Proctoring-Framework Progress

<b>NOTE : </b> Use ✅ for marking the progress

<details>
  
  <summary>:zap: <strong>Proposed Features </strong> </summary>
 
  
- ✅ &nbsp; Mouth Tracking
- ✅ &nbsp; Eye Tracking
- ✅ &nbsp; Gaze Detection
- ✅ &nbsp; Liveliness Detection
- ✅ &nbsp; Object Detection
- ❌ &nbsp; Voice Proctoring
 
</details>


### 🎆 Implemented Features

<details>
  
  <summary>:zap: <strong>Demonstrations</strong> </summary>

<br>
<p align="center">
 <kbd>
 <img height="400" width="700" src="https://user-images.githubusercontent.com/54600788/150630492-9ca34acc-7249-4cc0-8243-cddd284238eb.png"></img>
 </kbd>
</p>
<h4 align="center">Fig 1 : Face Localization & 68 Facial Landmark Mapping </h4>
<br><br>
<p align="center">
 <kbd>
 <img height="400" width="700" src="https://github.com/deepanshu-Raj/AI-Proctoring-Framework/blob/master/Videos/blink.gif"></img>
 </kbd>
</p>
<h4 align="center">Fig 2 : Blink Detection </h4>
<br><br>
<p align="center">
 <kbd>
 <img height="400" width="700" src="https://github.com/deepanshu-Raj/AI-Proctoring-Framework/blob/master/Videos/gaze.gif"></img>
 </kbd>
</p>
<h4 align="center">Fig 3 : Gaze Detection </h4>
<br><br>
<p align="center">
 <kbd>
 <img height="300" width="600" src="https://user-images.githubusercontent.com/54600788/150631473-6347b2a2-5b16-489e-8bb3-bf0c887a0786.png"></img>
 </kbd>
</p>
<h4 align="center">Fig 4 : Object Detection </h4>
<br><br>
<p align="center">
 <kbd>
 <img height="400" width="700" src="https://user-images.githubusercontent.com/54600788/150631583-42bd3479-4569-41aa-adbc-53f5d93d3510.png"></img>
 </kbd>
</p>
<h4 align="center">Fig 5 : Liveness Detection </h4>
<br><br>

</details>
