import cv2 as cv
import numpy as np
import pytesseract as tess

tess.pytesseract.tesseract_cmd = '/bin/tesseract' 
img=cv.imread("/home/sidharth/Documents/verify_id/app/data/college_id/LVtmoNLohm.jpg")
gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)

thresh = cv.adaptiveThreshold(gray,120,cv.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv.THRESH_BINARY,11,2)
cv.imshow("s",thresh)
cv.waitKey(0)
