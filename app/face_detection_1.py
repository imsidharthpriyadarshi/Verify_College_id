import cv2 as cv
import os
import time
path="/home/sidharth/Documents/verify_id/app/data/seg_train/potrait"
face_classifier = cv.CascadeClassifier(
        cv.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
i=0
for file in os.listdir(path=path):

    img=cv.imread(os.path.join(path, file))
    gray_image=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blurred_img = cv.GaussianBlur(img, (9, 9), 0)

    

    face = face_classifier.detectMultiScale(
        blurred_img, scaleFactor=1.1, minNeighbors=5, minSize=(200, 200)
    )
    for (x, y, w, h) in face:
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)
    i+=1
    print(i)
    cv.namedWindow("Resized_Window",cv.WINDOW_NORMAL)
    cv.resizeWindow("Resized_Window", 300, 700) 
    cv.imshow("Resized_Window", img)  
    cv.waitKey(3)  
    time.sleep(3)
   