# Project 1 Virtual Paint

##################################Steps################################################################################
# Step = 1, We have to enbale the webcam and check whether its working or not
# Step = 2, Using Color_picker tool decide the Max and Min Hue, Sat and Val values and store them mycolor list
# It will let us know the region where we have to define the contour
# Step = 3, We make a copy of our original img to draw on. We have to make a function that draw the contour aroung our mask and return the Pen nib coordinates
# After that we call getContour function inside the 'findcolor' function and draw a filled circle at the location
# of pen nib point
# We store the different coordinates of pen nib points in a list called myPoints as
# These points will be return value of our findColor function
# Step = 4 Define a drawOnCanvas function to drwa the different points on the canvas excrated from myPoints list

# Importing Libraries
import cv2
import numpy as np

# Setting Webcame Window
cap = cv2.VideoCapture(0)  # 0 is the id for defualt camera
cap.set(3,640)             # 3 is the id for width
cap.set(4,480)             # 4 is the id for heigth
cap.set(10,100)            # 10 is the id for brightness

# List of color_picker output values [[h_min,s_min,v_min,h_max,s_max,v_max]]
mycoloros = [[76, 172, 40, 179, 255, 255]]

# Empty list of points that will be drawn on the canvas
mypoints = []   #[x,y,]

# Function for creating mask and returing the pen nib coordinats using contor function
def findcolor(img,mycolors):
    imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)            # Converting Img to HSV
    newPoints = []
    lower = np.array(mycolors[0][0:3])                      # Lower bound values of Hue, Sat, Val
    upper = np.array(mycolors[0][3:6])                      # Upper bound values of Hue, Sat, Val
    mask = cv2.inRange(imgHSV, lower, upper)
    x,y = getContours(mask)                                 # Cordinates of Pen tip
    cv2.circle(imgResult,(x,y),10,(255,255,0),cv2.FILLED)   # Drawing Filled Circle on the pen nib
    if x!=0 and y!= 0:
        newPoints.append([x,y])                             # Appending the nib coordinates to list
    return newPoints
    #cv2.imshow("Mask",mask)

def getContours(img):
    contours, hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    x,y,w,h = 0,0,0,0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area>500:
            cv2.drawContours(imgResult,cnt,-1,(0,0,255),2) # Drawing Results on the img 'ImgResults'
            perimeter  = cv2.arcLength(cnt, True)          # True means our geometry is closed

            # Bounding box around shapes
            approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
            x, y, w, h = cv2.boundingRect(approx)          # Returning the origin coordinates and width and height of the contour
            # will return the nib coordinates
    return x+w//2, y

# Function to draw the points as circle on the canvas
def drawOnCanvas(mypoints):
    for point in mypoints:
        cv2.circle(imgResult, (point[0], point[1]), 10, (255, 255, 0), cv2.FILLED)

while True:
    success ,img = cap.read()                   # Reading the image using webcam, this image is used to create mask
                                                # Success return a boolean whether the images is captured or not
    img = cv2.flip(img, 1)                      # Inverting the camera
    imgResult = img.copy()                      # Making a copy of the original image to draw the results
    newPoints = findcolor(img, mycoloros)       # Calling FindColor function to get the pen nib coordinates each time we move the pen
    if len(newPoints)!=0:
        for newP in newPoints:
            mypoints.append(newP)
    if len(mypoints)!=0:
        drawOnCanvas(mypoints)                  # Drawing the points
    cv2.imshow("webcam_video",imgResult)        # Showing the results in real time

    if cv2.waitKey(1) and 0xFF == ord('q'):
        break
