
# Project 3 - Car Number Plate Detection
# We will cascade methods to detect the number plates

import cv2
import numpy as np

# Setting the webcam
############################################
widthImg = 480
heightImg = 640
casacade_path = "C:\\Users\\heman\\PycharmProjects\\OpenCvPython\\Resources\\haarcascade_russian_plate_number.xml"
NumPlateCascade = cv2.CascadeClassifier(casacade_path)
minArea = 500
color = (255,255,0)
############################################

# Setting Webcam
cap = cv2.VideoCapture(1)
cap.set(3, widthImg)
cap.set(4, heightImg)
cap.set(10,150)

while True:
    success , img = cap.read()
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    numberPlate = NumPlateCascade.detectMultiScale(img,1.1,4)

    for (x,y,w,h) in numberPlate:
        area = w*h
        if area > minArea:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),3)
            cv2.putText(img,"Number Plate",(x,y-5),cv2.FONT_HERSHEY_PLAIN,1,color,2)
            imgResult = img[y:y+h,x:x+w]
            cv2.imshow("webcam_video",imgResult)
    cv2.imshow("Image",img)
    #Success return a boolean whether the images is captured or not
    if cv2.waitKey(1) and 0xFF == ord('q'):
        break