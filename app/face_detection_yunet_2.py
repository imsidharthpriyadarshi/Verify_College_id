import cv2
import os
import time
import numpy as np
more_faces=[]
path="/home/sidharth/Documents/verify_id/app/data/seg_train/potrait"
detector = cv2.FaceDetectorYN.create("/home/sidharth/Documents/verify_id/app/face_detection_yunet_2023mar.onnx",  "", (320, 320),score_threshold=0.75)
total=0
det=0
for file in os.listdir(path):
    total+=1
    image_cv2_yunet = cv2.imread(os.path.join(path,file))
    height, width, _ = image_cv2_yunet.shape
    center=(height/2, width/2)
    detector.setInputSize((width, height))
    
    gray=cv2.cvtColor(image_cv2_yunet, cv2.COLOR_BGR2GRAY)
    #img1=cv2.convertScaleAbs(image_cv2_yunet)
    
    blurred_img = cv2.GaussianBlur(image_cv2_yunet, (3, 3), 0)
    
  

    
   



    cv2.namedWindow("Resized_Window",cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Resized_Window", 1020, 700) 
    #cv2.imshow("Resized_Window", blurred_img)
    #cv2.waitKey(3)  


    _, faces = detector.detect(blurred_img)

# if faces[1] is None, no face found
    
    if faces is not None:
        
        det+=1
        for face in faces:
        # parameters: x1, y1, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm
        
        # bouding box
            box = list(map(int, face[:4]))
            #print(box)
            color = (0, 0, 255)
            start=(box[0],box[1])
            end=(box[0]+box[2], box[1]+box[3])
            cx,cy=((start[0]+end[0])/2),((start[1]+end[1])/2)
            cv2.rectangle(image_cv2_yunet,start,end,color, 5)
            
            # confidence
            confidence = face[-1]
            confidence = "{:.2f}".format(confidence)
            position = (box[0], box[1] - 10)

            cv2.putText(image_cv2_yunet, confidence, position, cv2.FONT_HERSHEY_SIMPLEX, 3, color, 3, cv2.LINE_AA)
        if(len(faces)>1):
            cv2.namedWindow("Resized_Window",cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Resized_Window", 1020, 700) 
            cv2.imshow("Resized_Window", image_cv2_yunet)
            cv2.waitKey(5)
            time.sleep(5)
            #more_faces.append(os.path.join(path, file))
            print(os.path.join(path, file))    
    cv2.namedWindow("Resized_Window",cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Resized_Window", 1020, 700) 
    #cv2.imshow("Resized_Window", image_cv2_yunet)
    #cv2.waitKey(3)
    #time.sleep(3)
    #print(total, det)

    