import cv2 as cv
import tensorflow as tf
import numpy as np

class_name=['mountain', 'street', 'glacier', 'buildings', 'sea', 'forest']
class_name_lebel={class_name:i for i, class_name in enumerate(class_name)}
nb_classes=len(class_name)
print(class_name_lebel)
IMAGE_SIZE=(150,150)

import os 

def load_image():
    DIRECTORY="/home/sidharth/Documents/archive"
    CATEGORY=["seg_train", "seg_test"]
    output=[]
    for category in CATEGORY:
        path=os.path.join(os.path.join(DIRECTORY,category),category)
        images=[]
        labels=[]
        print("Loading {}".format(category))
        for folder in os.listdir(path):
            for file in os.listdir(os.path.join(path, folder)):
                print(file)
                img_path=os.path.join(os.path.join(path, folder), file)
                image=cv.imread(img_path)
                image=cv.cvtColor(image, cv.COLOR_BGR2RGB)
                image=cv.resize(image, IMAGE_SIZE)
                
