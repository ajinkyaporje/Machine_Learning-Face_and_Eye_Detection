# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 11:35:42 2019

@author: AJINKYA
"""
#importing Library
import cv2

"""
The files used for face detection are generally .xml files and are generated using a cascade trainer.
The cascade trainer takes input in form of positive and negative images,
the positves consist of the grayscale images that the machine needs to train itself for detection,
whereas the negatives consist of some random images that are used to train the machine against the positives,

"""
#importing files for face and eye detection.
"""
here face_cascade is a variable generated that stores the data from frontalface_cascade used for face detection,
similarly eye_cascade has an input file used for eyes detection.
"""
face_cascade= cv2.CascadeClassifier("F:\python downloads\images\images\haarcascade_frontalface_default.xml")
eye_cascade=cv2.CascadeClassifier("F:\python downloads\images\images\haarcascade_eye.xml")


def detect (gray,frame):
    faces= face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces: #for each face detected
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2) #we point a rectangle
        roi_gray = gray[y:y+h, x:x+w] #we get region of interest in the black and white format
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
        for (ex, ey, ew, eh) in eyes: #for each eye detected
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    return frame #we return the image with detector rectangles

video_capture =cv2.VideoCapture(0) #we turn on the web cam

while True: #we repeat the loop infinitely
    _, frame = video_capture.read() #we get the last frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #we do the same color transformations
    canvas = detect (gray, frame) #we get output of our detect function
    cv2.imshow('Video', canvas) #we display outputs
    if cv2.waitKey(1) & 0xFF == ord('q'): #if we type on keyboard: 'q' the loop would be interrupted.
        break #stop the loop
        
video_capture.release()
cv2.destroyAllWindows()
    