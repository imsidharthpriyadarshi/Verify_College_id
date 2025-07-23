import cv2 as cv
from ultralytics import YOLO
import time
import os
import shutil

path="/home/sidharth/Documents/verify_id/app/data/seg_test/landscape"
l_path="/home/sidharth/Documents/verify_id/app/data/seg_test/left_land"
r_path="/home/sidharth/Documents/verify_id/app/data/seg_test/right_land"


facemodel=YOLO("/home/sidharth/Documents/verify_id/app/yolov8n-face.pt")
total=0
det=0
for file in os.listdir(path):
    img= cv.imread(os.path.join(path, file))
    total+=1
    height, weight,_=img.shape
    print(height, weight)
    x_cen=height/2
    y_cen=weight/2
    img=cv.GaussianBlur(img, (3,3),0)
    face_result=facemodel.predict(img,conf=0.1)
    if face_result is not None:
        det+=1
        if len(face_result)>1:
            continue
        for info in face_result:
            parameters= info.boxes
            for box in parameters:
                x1, y1, x2, y2=box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                h,w = y2-y1,x2-x1
                cv.rectangle(img, (x1,y1),(x2,y2),(0,0,255),3)
                photo_xc=(y1+(h/2))
                photo_yc=(x1+(w/2))
                print(photo_yc)
                if y_cen+50<photo_yc:
                    try:
                        shutil.move(os.path.join(path, file),os.path.join(r_path, file))
                        print("right")
                    except:
                        print("Exception")    
                elif y_cen-50>photo_yc:
                    try:
                        shutil.move(os.path.join(path, file),os.path.join(l_path, file))
                        print("left")
                    except:
                        print("Exception")    



        #if(len(face_result)>1):
        #    cv.namedWindow("Resized_Window",cv.WINDOW_NORMAL)
        #    cv.resizeWindow("Resized_Window", 1020, 700) 
        #    cv.imshow("Resized_Window", img)
        #    cv.waitKey(5)
        #    time.sleep(5)
            #more_faces.append(os.path.join(path, file))
            #print(os.path.join(path, file))        
    cv.namedWindow("Resized_Window",cv.WINDOW_NORMAL)
    cv.resizeWindow("Resized_Window", 1020, 700) 
    #print(total, det)
    #cv.imshow("Resized_Window", img)
    #cv.waitKey(1)
    #time.sleep(1)

        