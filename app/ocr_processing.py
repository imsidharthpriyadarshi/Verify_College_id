import cv2 as cv
import tensorflow as tf
import numpy as np
import pytesseract as tess
from thefuzz import fuzz
from torchvision import transforms
import os
import Models
from scipy import ndimage


rotation_test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
classification_test_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

tess.pytesseract.tesseract_cmd = '/bin/tesseract'

libray_token=["library", "librarian","member","books"]
reciept_token=["invoice" ,"receipt", "fees" , "fee" , "transaction" , "tuition" , "cash" , "cheque" , "money" , "charges" , "payment" , "accounts" , "paid" , "bank" , "charges"]

rotation_model= Models.Model("app/rotation_iterdrop49.pth", ['0', '180', '270', '90'],rotation_test_transform)
is_photo_model=Models.Model("app/classification_iterdrop49.pth", ['documents', 'personal'],classification_test_transform)


def isLibraryCard(text:str):
    for token in libray_token:
        if token in text.lower():
            print(token)
            return True
    return False


def isFeeReciept(text:str):
    for token in reciept_token:
        if token in text.lower():
            print(token)
            return True
    return False

def score(extacted, given):
     token_set_score = fuzz.token_set_ratio(extacted, given)
     return token_set_score


path="/home/sidharth/Documents/datasets/library/"
usr_name=""
coll_name=""

for file in os.listdir(path):
    img_path=os.path.join(path, file)
    rotation_angle=rotation_model.get_rotation(img_path)
    angle=rotation_angle[0]
    img= cv.imread(img_path)
    if angle!=0:
        if angle==90:
            img=ndimage.rotate(img, -90, mode="nearest")
        else:    
            img= ndimage.rotate(img, 360-angle, mode="nearest")
    print("Checking for Library/Reciept >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")        
    gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    thresh= cv.threshold(gray,90,255,cv.THRESH_BINARY)[1]
    median = cv.medianBlur(thresh, 7)
    text=tess.image_to_string(median)
    print(img_path)
    if isLibraryCard(text):
        print("library")
    elif isFeeReciept(text):
        print("receipt")
    else:
         print("other") 

         if is_photo_model[0]=="personal":
            print("personal")
         else:
             if score(text, usr_name)>=70:
                 if score(text, coll_name)>=70:
                     print("datas matched")
                 else:
                     print("Name matched & college name haven't matched")    
             else:
                 print("Name not matched")             
            




#before coming here first check rotation of image
#then put personal_photo label to the all personal photo