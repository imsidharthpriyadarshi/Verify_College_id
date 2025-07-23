import torch
import cv2

# Download model from github
model = torch.hub.load('ultralytics/yolov5', 'yolov5n')
    
img = cv2.imread("/home/sidharth/Documents/verify_id/app/rotations_data/0aDBJDO9rE.jpg")

# Perform detection on image
result = model(img)
print('result: ', result)