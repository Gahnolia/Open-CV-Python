import cv2
import numpy as np
print('Successfully imported')

# Lesson 1 covers the topic related to importing 'Images' , 'Videos' and 'Webcam'
# Part 4 -Important fuctions
# Part 5 -  Resizing and Cropping
# Part 6 -  Shapes and texts
# Part 7 -  Warp Prespective
# Part 8 - Joining Images
#Part 9 - Color Detection
#Part 10 - Contours and Shape Detection
# Part 11 - Face detection

##############################################################################################
# Part 1 - Reading images
#img_path = "C:\\Users\\heman\\PycharmProjects\\OpenCvPython\\Resources\lena.png"
# img = cv2.imread(img_path)
#
# #Displaying Image
# cv2.imshow('Output_image',img)
# cv2.waitKey(0)
###############################################################################################

###############################################################################################
# Part 2 - Dealing with videos
# cap = cv2.VideoCapture(Path_video)
#
#  while True:
#      success , img = cap.read()
#      cv2.imshow("webcam_video",img)
# #     #Success return a boolean whether the images is captured or not
#      if cv2.waitKey(1) and 0xFF ==q ord('q'):
#          break
         # this if statement is adding delay in the video and if we press 'q' then the video will stop
###############################################################################################


###############################################################################################
#Using webcam
cap = cv2.VideoCapture(0)  # 0 is the id for defualt camera
cap.set(3,640)             # 3 is the id for width
cap.set(4,480)             # 4 is the id for heigth
cap.set(10,100)            # 10 is the id for brightness

while True:
    success , img = cap.read()
    cv2.imshow("webcam_video",img)
    #Success return a boolean whether the images is captured or not
    if cv2.waitKey(1) and 0xFF == ord('q'):
        break
###############################################################################################

###############################################################################################
# Part - 4 Using Fuctons
#
# #Converting to Grey Scale
# img = cv2.imread(img_path)
# im_grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# cv2.imshow('Grey_Image',im_grey)
# #cv2.waitKey(0)
#
# #Blur image
# im_blur = cv2.GaussianBlur(im_grey,(7,7),0)
# cv2.imshow('Blur_Image',im_blur)
# #cv2.waitKey(0)
#
# #Edge Detector
# img_canny = cv2.Canny(img,100,100)
# cv2.imshow('Canny_Image',img_canny)
# #cv2.waitKey(0)
#
# #Increasing the thickness of edge
# kernel = np.ones((5,5),np.uint8)
# imgDialation = cv2.dilate(img_canny,kernel, iterations=1)
# cv2.imshow('Dialation_Image',imgDialation)
# #cv2.waitKey(0)
#
# #Opposite of Dialation-> making the edge thiner
# imgEroded = cv2.erode(imgDialation,kernel,iterations=1)
# cv2.imshow('Eroded_Image',imgEroded)
# cv2.waitKey(0)

###############################################################################################

# Part 5 -  Resizing and Cropping
# Resizing
# img_path_2 = "C:\\Users\\heman\\PycharmProjects\\OpenCvPython\\Resources\\lambo.png"
# img = cv2.imread(img_path_2)
# print(img.shape) # Return (height, width, channel)
# img_resize = cv2.resize(img,(1200,1400))  #Argument as (Width, height)
# print(img_resize.shape)
# cv2.imshow("lambo",img)
# cv2.imshow("lambo_resized",img_resize)
# # cv2.waitKey()
#
# # Cropping
# imgCropped = img[0:200,200:500] # {(height, Width)}
# cv2.imshow("lambo_cropped",imgCropped)
# cv2.waitKey(0)

###############################################################################################
# Part 6 -  Shapes and texts
# img = np.zeros((512,512,3),np.uint8)
# #img[:] = 255,0,0 #Blue, Green, Red
#
# # Lines
# cv2.line(img,(0,0),(img.shape[1],img.shape[0]),(0,255,0),2)
#
# # rectangle
# cv2.rectangle(img,(0,0),(250,400),(0,0,250),3)
# #cv2.rectangle(img,(0,0),(250,400),(0,0,250),cv2.FILLED)
#
# # Circle
# cv2.circle(img,(200,200),25,(255,255,0),2)
#
# # Text
# cv2.putText(img,"OpenCV",(345,200),cv2.FONT_HERSHEY_SIMPLEX,1,(0,150,0),3)
# cv2.imshow("Image",img)
# cv2.waitKey(0)

###############################################################################################
# Part 7 - Warp Prespective
# img_path_3 = "C:\\Users\\heman\\PycharmProjects\\OpenCvPython\\Resources\\cards.jpg"
# img = cv2.imread(img_path_3)
#
# width, heigth = 250,350
# points1 = np.float32([[111,219],[287,188],[154,482],[352,440]])
# points2 = np.float32([[0,0],[width,0],[0,heigth],[width,heigth]])
# matrix = cv2.getPerspectiveTransform(points1,points2)
# imgOutput = cv2.warpPerspective(img,matrix,(width,heigth))
#
# cv2.imshow("Cards",img)
# cv2.imshow("Wrapped_Cards",imgOutput)
# cv2.waitKey(0)

###############################################################################################
# Part 8 - Joining Images

# img = cv2.imread(img_path)
#
# img_hor = np.hstack((img,img))
# img_var = np.vstack((img,img))
#
# # for using these method the images have to be same size(matrix)
# # otherwise this method wont work
# cv2.imshow("Horizonatal", img_hor)
# cv2.imshow("Vertical", img_var)
# cv2.waitKey()

# # Function to deal with different sizes of images
# def stackImages(scale,imgArray):
#     rows = len(imgArray)
#     cols = len(imgArray[0])
#     rowsAvailable = isinstance(imgArray[0], list)
#     width = imgArray[0][0].shape[1]
#     height = imgArray[0][0].shape[0]
#     if rowsAvailable:
#         for x in range ( 0, rows):
#             for y in range(0, cols):
#                 if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
#                     imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
#                 else:
#                     imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
#                 if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
#         imageBlank = np.zeros((height, width, 3), np.uint8)
#         hor = [imageBlank]*rows
#         hor_con = [imageBlank]*rows
#         for x in range(0, rows):
#             hor[x] = np.hstack(imgArray[x])
#         ver = np.vstack(hor)
#     else:
#         for x in range(0, rows):
#             if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
#                 imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
#             else:
#                 imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
#             if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
#         hor= np.hstack(imgArray)
#         ver = hor
#     return ver
#
# img = cv2.imread(img_path)
# img_grey = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
# stack_Image = stackImages(0.5,([img,img_grey,img],[img,img,img_grey]))
# cv2.imshow("Stacked_Img",stack_Image)
# cv2.waitKey()

###############################################################################################

#Part 9 - Color Detection

# def empty(a):
#     pass
#
# img_path_2 = "C:\\Users\\heman\\PycharmProjects\\OpenCvPython\\Resources\\lambo.png"
# cv2.namedWindow("TrackBars")
# cv2.resizeWindow("TrackBars",640,240)
# cv2.createTrackbar("Hue Min","TrackBars",0,179,empty)
# cv2.createTrackbar("Hue Max","TrackBars",179,179,empty)
# cv2.createTrackbar("Sat Min","TrackBars",0,255,empty)
# cv2.createTrackbar("Sat Max","TrackBars",255,255,empty)
# cv2.createTrackbar("val Min","TrackBars",0,255,empty)
# cv2.createTrackbar("val Max","TrackBars",255,255,empty)
#
# while True:
#     img = cv2.imread(img_path_2)
#     imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     h_min = cv2.getTrackbarPos("Hue Min", "TrackBars")
#     h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
#     s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
#     s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
#     v_min = cv2.getTrackbarPos("val Min", "TrackBars")
#     v_max = cv2.getTrackbarPos("val Max", "TrackBars")
#
#     lower = np.array([h_min,s_min,v_min])
#     upper = np.array([h_max,s_max,v_max])
#     mask  = cv2.inRange(imgHSV,lower,upper)
#     imgResult = cv2.bitwise_and(img,img,mask= mask)
#
#     # cv2.imshow("original", img)
#     # cv2.imshow('HSV', imgHSV)
#     # cv2.imshow("mask",mask)
#     # cv2.imshow("ImgResult",imgResult)
#     stackedImg = stackImages(0.5,([img,imgHSV],[mask,imgResult]))
#     cv2.imshow("Stacked Images",stackedImg)
#     cv2.waitKey(1)

###############################################################################################
#Part 10 - Contours and Shape Detection

# def getContours(img):
#     contours, hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
#     for cnt in contours:
#         area = cv2.contourArea(cnt)
#         if area>500:
#             cv2.drawContours(imgContour,cnt,-1,(255,0,0),2)
#             perimeter  = cv2.arcLength(cnt, True) # True means our geometry is closed
#             # Number of Edges
#             approx = cv2.approxPolyDP(cnt,0.02*perimeter,True)
#             objCorner = len(approx)
#
#             # Categorising the shapes
#             if objCorner ==3:
#                 objectType = 'Tri'
#             elif objCorner == 4:
#                 aspectRatio = w/float(h)
#                 if aspectRatio >0.95 and aspectRatio < 1.05:
#                     objectType = "Rectangle"
#                 else:
#                     objectType = "Square"
#             elif objCorner >4:
#                 objectType = 'Cirlce'
#
#             else: objectType = 'None'
#
#             # Bounding box around shapes
#             x, y, w, h = cv2.boundingRect(approx)
#             # x y are the origin coordinates and w, h are width and height respectively
#             cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 3)
#             cv2.putText(imgContour,objectType,((x+(w//2)-10),(y+(h//2)-10)),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),2)
#
# img_path_4 = "C:\\Users\\heman\\PycharmProjects\\OpenCvPython\\Resources\\shapes.png"
# img = cv2.imread(img_path_4)
# imgContour = img.copy()
# img_grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# imgBlur  = cv2.GaussianBlur(img_grey,(7,7),1)
#
# # we will detected the edges in the images
# imgCanny = cv2.Canny(imgBlur,50,50)
#
# imgBlank = np.zeros_like(img)
#
# getContours(imgCanny)
# stackedImg = stackImages(0.6,([img,img_grey,imgBlur],[imgCanny,imgContour,imgBlank]))
# cv2.imshow(" Stacked Images",stackedImg)
# cv2.waitKey()

###############################################################################################
# Part 11 - Face detection
# we will cascade file to detect the face
# cascade are pre trained model

# casacade_path = "C:\\Users\\heman\\PycharmProjects\\OpenCvPython\\Resources\\haarcascade_frontalface_alt.xml"
# faceCascade = cv2.CascadeClassifier(casacade_path)
#
# img = cv2.imread(img_path)
# #imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#
# faces = faceCascade.detectMultiScale(img,1.1,4)
#
# for (x,y,w,h) in faces:
#     cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),3)
#
# cv2.imshow("Image",img)
# cv2.waitKey()