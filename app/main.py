import cv2 as cv
import numpy as np
import pytesseract as tess

tess.pytesseract.tesseract_cmd = '/bin/tesseract' 
img=cv.imread("/home/sidharth/Documents/verify_id/app/data/college_id/LRUWGkPtIM.jpg")
gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)

thresh = cv.adaptiveThreshold(gray,256,cv.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv.THRESH_BINARY,11,2)
"""cv.namedWindow("Resized_Window", cv.WINDOW_NORMAL) 
edges = cv.Canny(thresh,100,200)  
# Using resizeWindow() 
cv.resizeWindow("Resized_Window", 300, 700) 
cv.imshow("Resized_Window",edges)
cv.waitKey(0)"""
text=tess.image_to_string(gray)
print(text)
