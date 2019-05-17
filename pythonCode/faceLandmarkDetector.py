import cv2
import dlib
import sys
import numpy as np

#draw landmark points on the image
def drawPoints(image, landmarks):
    for part in landmarks.parts():
        cv2.circle(image, (part.x, part.y), 3, (0, 255, 255), -1)

#define face detector
faceDetector = dlib.get_frontal_face_detector()

#define landmark detector and load the landmark model
landmarkDetector = dlib.shape_predictor("../dlibAndModel/shape_predictor_68_face_landmarks.dat")

#read images with single and multiple persons
imageSingle = cv2.imread("../assets/anish.jpg")
imageMultiple = cv2.imread("../assets/anish2.jpg")

#create images clone to woek on
imageSingleClone = imageSingle.copy()
imageMultipleClone = imageMultiple.copy()

#convert image to dlibs image format
dlibSingleImage = cv2.cvtColor(imageSingleClone, cv2.COLOR_BGR2RGB)
dlibMultipleImage = cv2.cvtColor(imageMultipleClone, cv2.COLOR_BGR2RGB)

#detect faces in the image
facesSingle = faceDetector(dlibSingleImage, 0)
facesMultiple = faceDetector(dlibMultipleImage, 0)

#loop over all the faces detected
for i in range(0, len(facesSingle)):
    dlibRect = dlib.rectangle(int(facesSingle[i].left()), int(facesSingle[i].top()), int(facesSingle[i].right()), int(facesSingle[i].bottom()))
    #for each face run landmark detector
    landmarks = landmarkDetector(dlibSingleImage, dlibRect)
    #Print number of landmark point detected
    print("Number of landmark points detected: ", len(landmarks.parts()))
    #draw landmarks on the face
    drawPoints(imageSingleClone, landmarks)

for i in range(0, len(facesMultiple)):
    dlibRect = dlib.rectangle(int(facesMultiple[i].left()), int(facesMultiple[i].top()), int(facesMultiple[i].right()), int(facesMultiple[i].bottom()))
    #for each face run landmark detector
    landmarks = landmarkDetector(dlibMultipleImage, dlibRect)
    #Print number of landmark point detected
    print("Number of landmark points detected: ", len(landmarks.parts()))
    #draw landmarks on the face
    drawPoints(imageMultipleClone, landmarks)

#create windows to display images
cv2.namedWindow("single person image", cv2.WINDOW_NORMAL)
cv2.namedWindow("single person landmarks", cv2.WINDOW_NORMAL)
cv2.namedWindow("multiple person image", cv2.WINDOW_NORMAL)
cv2.namedWindow("multiple person landmarks", cv2.WINDOW_NORMAL)

#display images
cv2.imshow("single person image", imageSingle)
cv2.imshow("single person landmarks", imageSingleClone)
cv2.imshow("multiple person image", imageMultiple)
cv2.imshow("multiple person landmarks", imageMultipleClone)

#press esc to exit the program
cv2.waitKey(0)

#close all the opened windows
cv2.destroyAllWindows()