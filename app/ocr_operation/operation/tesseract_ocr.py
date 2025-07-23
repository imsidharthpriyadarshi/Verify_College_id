import pytesseract as tess
import cv2 as cv

tess.pytesseract.tesseract_cmd = '/bin/tesseract'

def extract(img_path):
    img= cv.imread(img_path)
    img= cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    text=tess.image_to_string(img)
    return text
