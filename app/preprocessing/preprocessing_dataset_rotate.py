from scipy import ndimage
import cv2 as cv
import os
import shutil
"""
cv.namedWindow("resize", cv.WINDOW_NORMAL)
cv.resizeWindow("resize",720, 400)"""
path="/home/sidharth/Verify_College_id/app/data"
path_90="/home/sidharth/Verify_College_id/app/rotation_data/90"
path_180="/home/sidharth/Verify_College_id/app/rotation_data/180"
path_270="/home/sidharth/Verify_College_id/app/rotation_data/270"
path_0="/home/sidharth/Verify_College_id/app/rotation_data/0"
i =0
#cv.imshow("resize", image)
for file in os.listdir(path):
    file_path=os.path.join(path, file)
    image=cv.imread(file_path)
    height, widht, _=image.shape
    shutil.move(os.path.join(path, file),os.path.join(path_0, file))
    rotated_90 = ndimage.rotate(image,90,mode="nearest")
    cv.imwrite(os.path.join(path_90,file),rotated_90)
    rotated_180 = ndimage.rotate(image, 180, mode="nearest")
    cv.imwrite(os.path.join(path_180,file),rotated_180)
    rotated_270 = ndimage.rotate(image, 270, mode="nearest")
    cv.imwrite(os.path.join(path_270,file),rotated_270)
    i+=1
    print(i)
    if i==10:
        break

    

