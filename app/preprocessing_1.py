import cv2 as cv
import os
import shutil
path="./app/data/seg_test/college_id"
pot="./app/data/seg_test/portrait"
land="./app/data/seg_test/landscape"

for file in os.listdir(path):
    img=cv.imread(os.path.join(path,file))
    height, width, channel=img.shape
    if height>width:
        shutil.move(os.path.join(path, file), os.path.join(pot, file))
        print("moved 1")
    elif height<width:
        shutil.move(os.path.join(path, file), os.path.join(land, file))    
        print("moved 2")


