import cv2 as cv
import numpy as np
import matplotlib as plt

face_cascade = cv.CascadeClassifier('/Users/amanda/opencv/data/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('/Users/amanda/opencv/data/haarcascades/haarcascade_eye.xml')


path = '/Users/amanda/Google\\ Drive/School/USF/PhD/Major_Research/Lie_detection/MU3D-Package/Images/BF001_1PT.png'
img = cv.imread(path)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('gray image',img)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)

for (x,y,w,h) in faces:
        # To draw a rectangle in a face
        cv.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        # Detects eyes of different sizes in the input image
        eyes = eye_cascade.detectMultiScale(roi_gray)

        #To draw a rectangle in eyes
        for (ex,ey,ew,eh) in eyes:
            cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,127,255),2)





window_name = 'image'
cv.imshow(window_name,img)
cv.waitKey(0)
cv.destroyAllWindows()
