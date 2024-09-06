import cv2 as cv
import numpy as np
import pytesseract as tess

tess.pytesseract.tesseract_cmd = '/bin/tesseract' 
img=cv.imread("/home/sidharth/Documents/verify_id/app/id2.jpg")
gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)

res,thresh= cv.threshold(gray,80,255,cv.THRESH_BINARY)

median = cv.medianBlur(thresh, 3)

text=tess.image_to_string(gray)
print(text)



