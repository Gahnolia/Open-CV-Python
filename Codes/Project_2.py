
# Project 2 - Document Scanning using Webcam

########### STEPS #########################
# Step 1 -  Setting the webcam
# Step 2 -  Preprocessing Image
# Step 3 -  Get the contours
# Step 4 -  Wraping perspective of image

import cv2
import numpy as np

# Setting the webcam
widthImg = 480
heightImg = 640
cap = cv2.VideoCapture(1)    # 1 Represents that we are using a different camera other than the inbuild webcam
cap.set(3, widthImg)
cap.set(4, heightImg)
cap.set(10,150)


def preProcessing(img):
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)                  # Changing the image to Gray Scale
    imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)                     # Applying Blur Filter
    imgCanny = cv2.Canny(imgBlur,200,200)                           # Detecting Edges
    kernel = np.ones((5,5))
    imgDialation = cv2.dilate(imgCanny,kernel,iterations=2)         # Making the edge thicker
    imgThreshold = cv2.erode(imgDialation,kernel,iterations=1)      # Making the edge thiner
    # Returning the fully preprocessed image
    return imgThreshold


def getContours(img):
    contours, hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    biggest = np.array([])
    maxArea = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area>5000:
            cv2.drawContours(imgContour,cnt,-1,(0,0,255),2)     # Drawing Results on the img 'ImgContour'
            perimeter  = cv2.arcLength(cnt, True)               # True means our geometry is closed
            # Bounding box around shapes
            approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)  # It returns a set of (x,y) points
            if area > maxArea and len(approx) == 4:                 # since we are looking for rectangular page we must have 4 corner points
                biggest = approx                                    # if there are many rectangular objects then we are selecting a object with largest area
                maxArea = area
    # Return the 4 points as biggest
    return biggest

def getWrap(img,biggest):
    #print(biggest.shape)
    biggest = reorder(biggest)                                      # Calliing the reorder function to reorder the indexig of 4 corner points
    points1 = np.float32(biggest)                                   # We want them in the format of ([[0,0],[widthImg,0],[0,heightImg],[widthImg,heightImg]])
    points2 = np.float32([[0,0],[widthImg,0],[0,heightImg],[widthImg,heightImg]])
    matrix = cv2.getPerspectiveTransform(points1,points2)
    imgOutput = cv2.warpPerspective(img,matrix,(widthImg,heightImg))

    return imgOutput

def reorder(myPoints):
    myPoints = myPoints.reshape((4,2))              # Biggest is shape of (4,1,2) where 1 is redundant so we reshaped Biggest
    myPointsNew = np.zeros((4,1,2),np.int32)
    add = myPoints.sum(1)                           # Here we applied the logic that if we add the pair in the list, so min sum will give (0,0) point and max sum will give [widthImg,heightImg]
    diff = np.diff(myPoints,axis = 1)               # Similarly if we subtract the pair in the list, so min diff will give [widthImg,0] point and max diff will give [0,heightImg]
    myPointsNew[0] = myPoints[np.argmin(add)]       # Reassigning the points in correct order into the myPoints list
    myPointsNew[3] = myPoints[np.argmax(add)]
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]

    return myPointsNew


# Function to get Stacked Images
def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

while True:
    success , img = cap.read()                      # Reading Images from Webcam
    img = cv2.flip(img,1)                           # Inverting the image
    img = cv2.resize(img,(widthImg,heightImg))      # Resizing the image
    imgContour = img.copy()                         # making a copy of img to plot the contours on
    imgThresho = preProcessing(img)                 # Calling PreProcessing Function
    biggest = getContours(imgThresho)               # Getting the 4 Points in correct ordered format
    if biggest.size != 0 :                          # Applying condition that if we dont find 4 points in image then we will simply retunr the original image
        imgWrapped = getWrap(img,biggest)
        imagearray = ([img,imgThresho],
                  [imgContour,imgWrapped])
    else:
        imagearray = ([img, imgThresho],
                      [img, img])

    stackedImg = stackImages(0.5,imagearray)       # Stacking the Images
    cv2.imshow("ImgThreshold",stackedImg)          # Displaying the stacking Images
    #Success return a boolean whether the images is captured or not
    if cv2.waitKey(1) and 0xFF == ord('q'):
        break
