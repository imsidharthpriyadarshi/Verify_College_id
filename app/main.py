import os

from fastapi import FastAPI,HTTPException, Query,Depends
from app.schemas import schemas
import cv2 as cv
from app.model import Models,Modelss
from app.preprocessing import transformations
from scipy import ndimage
import math
from typing import Optional
import requests
import shutil
from tempfile import NamedTemporaryFile
import numpy as np
from app.ocr_operation import ocr_operations
import shutil
from pydantic import HttpUrl
from google.cloud import vision_v1
from google.cloud.vision_v1 import types
import re
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] ='app/visions_key.json'
from google.cloud import vision
app= FastAPI()
libray_token=["library", "librarian","books"]
reciept_token=["invoice" ,"receipt", "fees" , "fee" , "transaction" , "tuition fees" , "cash" , "cheque" , "money" , "charges" , "payment" , "accounts" , "paid" , "bank" , "charges"]
department_variants = {
    "Aeronautical Engineering": ["AERO", "Aeronautics", "Aeronautical Engg"],
    "Aerospace Engineering": ["AE", "Aerospace Engg"],
    "Agricultural Engineering": ["AG", "Agri Engg"],
    "Architecture and Regional Planning": ["ARP", "Architecture", "Planning"],
    "Artificial Intelligence and Data Science": ["AIDS", "AI & DS", "AI and Data Science"],
    "Artificial Intelligence and Machine Learning": ["AIML", "AI & ML", "AI and ML"],
    "Automobile Engineering": ["AUTO", "Automobile Engg"],
    "BSc Research": ["BSCR", "B.Sc Research", "Bachelor of Science Research"],
    "Bachelor of Arts": ["BA", "Arts"],
    "Bio-Medical Engineering": ["BME", "Biomedical Engg"],
    "Bio-Technology": ["BT", "Biotech"],
    "Biology": ["BIO", "Bio"],
    "Biotechnology": ["BT", "Biotech", "Bio-Tech"],
    "Business Administration": ["BBA", "Business Admin"],
    "Ceramic and Industrial Design": ["CID", "Ceramic Design", "Industrial Design"],
    "Chemical Engineering": ["CHE", "Chemical Engg"],
    "Chemistry": ["CHEM", "Chem"],
    "Civil Engineering": ["CE", "Civil Engg"],
    "Commerce": ["COM", "B.Com", "Commerce Studies"],
    "Communication and Computer engineering": ["CCE", "Communication & Computer Engg",],
    "Computer Application": ["CA", "BCA", "Comp Application"],
    "Computer Engineering": ["CEng", "Computer Engg"],
    "Computer Science": ["CS", "Comp Sci"],
    "Computer Science and Engineering": ["CSE", "CS & Engg", "Comp Science and Engineering","Comp. Sci. & Engineering"],
    "Construction Technology": ["CT", "Construction Tech", "Const Tech"],
    "Data Science": ["DS", "DataSci"],
    "Economics": ["ECO", "Econ"],
    "Education": ["EDU", "Edu"],
    "Electrical Engineering": ["EE", "Electrical Engg"],
    "Electrical and Computer Engineering": ["ECE", "Electrical & Comp Engg"],
    "Electrical and Electronics Engineering": ["EEE", "Electrical & Electronics"],
    "Electronics And Computer Engineering": ["ECEng", "Electronics & Comp Engg"],
    "Electronics Engineering": ["EC", "Electronics Engg"],
    "Electronics Systems": ["ES", "Electronic Systems"],
    "Electronics and Communication Engineering": ["ECE", "Electronics & Comm Engg"],
    "Electronics and Instrumentation Engineering": ["EIE", "Electronics & Instrumentation"],
    "Electronics and Telecommunication Engineering": ["ETC", "Electronics & Telecomm"],
    "Energy Science and Engineering": ["ESE", "Energy Engineering"],
    "English": ["ENG", "English Studies", "Lit"],
    "Environmental Engineering": ["EN", "Environmental Engg"],
    "Fine Arts": ["FA", "Arts", "Fine Art"],
    "Geography": ["GEO", "Geog"],
    "History": ["HIS", "Hist"],
    "Industrial Electronics": ["IEE", "Industrial Elec"],
    "Industrial Engineering": ["IND", "Industrial Engg"],
    "Information Science": ["IS", "Info Sci"],
    "Information Technology": ["IT", "Info Tech"],
    "Information and Communication Technology": ["ICT", "Info & Comm Tech"],
    "Instrumentation Engineering": ["IE", "Instrumentation Engg"],
    "Instrumentation and Control Engineering": ["ICE", "Instrumentation & Control"],
    "Law": ["LAW", "LLB"],
    "Library Science": ["LIB", "Library Sci"],
    "Management": ["MGMT", "MBA", "Business Management"],
    "Marine Engineering": ["MAR", "Marine Engg"],
    "Material Science Engineering": ["MSE", "Material Sci Engg"],
    "Mathematics": ["MATH", "Maths"],
    "Mathematics and Computing": ["MAC", "Maths & Computing"],
    "Mechanical Engineering": ["ME", "Mech Engg"],
    "Mechanical and Automation Engineering": ["MAE", "Mechanical & Automation"],
    "Mechatronics": ["MTRX", "Mechatronics Engg"],
    "Medical Electronics": ["MED", "Medical Elec"],
    "Metallurgical Engineering": ["MT", "Metallurgical Engg"],
    "Metallurgy": ["MET", "Metallurgy Engg"],
    "Mining Engineering": ["MIN", "Mining Engg"],
    "Oil Technology": ["OIL", "Oil Tech"],
    "Others": ["OTH", "Other"],
    "Petroleum Engineering": ["PET", "Petroleum Engg"],
    "Pharmaceutical": ["PHARMA", "Pharma"],
    "Philosophy": ["PHIL", "Phil"],
    "Physical Education": ["PEd", "PE"],
    "Physics": ["PHY", "Phys"],
    "Physics/Chemistry/Mathematics": ["PCM", "PCMB", "Science PCM"],
    "Political Science": ["POL", "Pol Sci"],
    "Production Engineering": ["PE", "Production Engg"],
    "Psychology": ["PSY", "Psych"],
    "Renewable Energy": ["RE", "Renewable Energy Engg"],
    "Robotics and Artificial Intelligence": ["RAI", "Robotics & AI"],
    "Robotics and Automation": ["RA", "Robotics & Automation"],
    "Sociology": ["SOC", "Socio"],
    "Software Engineering": ["SE", "Soft Engg"],
    "Telecommunication Engineering": ["TELE", "Telecom Engg"],
    "Textile Engineering": ["TE", "Textile Engg"],
    "Textile Technology": ["TT", "Textile Tech"],
    "Urban Management": ["UM", "Urban Mgmt"]
}


rotation_model= Modelss.Modelss("/home/sid/Documents/Verify_College_id/rotnet_final1.pth", ['0', '180', '270', '90'],transformations.rotation_test_transform)
is_photo_model=Models.Model("app/model/trained/document_personal_classification_iterdrop49.pth", ['documents', 'personal'],transformations.classification_test_transform)
orientation_model=Models.Model("app/model/trained/orientation_classification_iterdrop49.pth",['landscape','portrait'],transformations.classification_test_transform)
landscape_verify_id_model=Models.Model("/home/sid/Documents/Verify_College_id/app/model/trained/land_id_classification_iterdrop49.pth",['id','other_document','others'],transformations.classification_test_transform)
portrait_verify_id_model=Models.Model("/home/sid/Documents/Verify_College_id/app/model/trained/por_id_classification_iterdrop49.pth",['id','other_document','others'],transformations.classification_test_transform)

@app.get("/")
def deployed():
    return {"is_working":True}
@app.post("/test")
def read_root(folder:schemas.Input):
    ii=0
    folder_name=folder.folder_name
    for file in os.listdir(folder_name):
        img_path= os.path.join(folder_name,file)
        #img= cv.imread(img_path)
        rotation_angle=rotation_model.get_rotation(img_path)
        angle=int(rotation_angle[0])
        accuracy=float(rotation_angle[1])
        if accuracy >= 0.99:
                if angle ==90:
                    # if accuracy !=1:
                    #     ii+=1
                    #     print(90, "continue")
                    #     continue
                    shutil.move(img_path, os.path.join("/home/sid/Documents/eYRC22_TeamPhoto/90",file))
                    print(90, "moved")
                    #img=ndimage.rotate(img, -90,mode="nearest")
                elif angle==180:
                    # if accuracy !=1:
                    #     ii+=1
                    #     print(180, "continue")
                    #     continue
                    shutil.move(img_path, os.path.join("/home/sid/Documents/eYRC22_TeamPhoto/180",file))
                    print(180, "moved")
                elif angle==270:
                    # if accuracy !=1:
                    #     ii+=1
                    #     print(270, "continue")
                    #     continue
                    shutil.move(img_path, os.path.join("/home/sid/Documents/eYRC22_TeamPhoto/270",file))
                    print(270, "moved")
                elif angle ==0:
                    shutil.move(img_path, os.path.join("/home/sid/Documents/eYRC22_TeamPhoto/0",file))
                    print(0, "moved")
                    #img=ndimage.rotate(img,360-angle, mode="nearest")
                print(angle,accuracy)    
                #cv.imwrite(img_path,img)
        else:
            ii+=1
            print(file, f"not confirm {ii}, {accuracy} ,{angle}")    
    return {"ddk":"dkk"}



# for document 0.90 or even 0.85 percent is fine but for personal photo it is fine even for above 0.50 percent
@app.post("/test_doc")
def read_root(folder:schemas.Input):
    ii=0
    folder_name=folder.folder_name
    for file in os.listdir(folder_name):
        img_path= os.path.join(folder_name,file)
        #img= cv.imread(img_path)
        rotation_angle=is_photo_model.get_rotation(img_path)
        result=rotation_angle[0]
        accuracy=float(rotation_angle[1])
        if accuracy >= 0.5:
            
                if result =="documents":
                    if accuracy >=0.85:
                        shutil.move(img_path, os.path.join("/home/sid/Documents/eYRC22_TeamPhoto/00_doc",file))
                    else:
                        ii+=1    
                    #img=ndimage.rotate(img, -90,mode="nearest")
                elif result=="personal":
                    shutil.move(img_path, os.path.join("/home/sid/Documents/eYRC22_TeamPhoto/00_personal",file))
                print(result,accuracy)    
                #cv.imwrite(img_path,img)
        else:
            ii+=1
            print(file, f"not confirm {ii}, {accuracy} ,{result}")    
    return {"ddk":"dkk"}



@app.post("/test_orientation")
def read_root(folder:schemas.Input):
    ii=0
    folder_name=folder.folder_name
    for file in os.listdir(folder_name):
        img_path= os.path.join(folder_name,file)
        #img= cv.imread(img_path)
        rotation_angle=orientation_model.get_rotation(img_path)
        result=rotation_angle[0]
        accuracy=float(rotation_angle[1])
        if accuracy >= 0.99:
                if result =="landscape":
                        shutil.move(img_path, os.path.join("/home/sid/Documents/eYRC22_TeamPhoto/test_landscape",file))   
                    #img=ndimage.rotate(img, -90,mode="nearest")
                elif result=="portrait":
                    shutil.move(img_path, os.path.join("/home/sid/Documents/eYRC22_TeamPhoto/test_portrait",file))
                print(result,accuracy)    
                #cv.imwrite(img_path,img)
        else:
            ii+=1
            print(file, f"not confirm {ii}, {accuracy} ,{result}")    
    return {"ddk":"dkk"}

@app.post("/test_land_verify_id")
def read_root(folder:schemas.Input):
    ii=0
    folder_name=folder.folder_name
    for file in os.listdir(folder_name):
        img_path= os.path.join(folder_name,file)
        #img= cv.imread(img_path)
        rotation_angle=landscape_verify_id_model.get_rotation(img_path)
        result=rotation_angle[0]
        accuracy=float(rotation_angle[1])
        if accuracy >= 0.90:
                if result =="id":
                        shutil.move(img_path, os.path.join("/home/sid/Documents/test_data/id",file))   
                    #img=ndimage.rotate(img, -90,mode="nearest")
                elif result=="other_document":
                    shutil.move(img_path, os.path.join("/home/sid/Documents/test_data/doc",file))
                elif result=="others":
                     shutil.move(img_path, os.path.join("/home/sid/Documents/test_data/other",file))    
                print(result,accuracy)    
                #cv.imwrite(img_path,img)
        else:
            ii+=1
            print(file, f"not confirm {ii}, {accuracy} ,{result}")    
    return {"ddk":"dkk"}
#this is just for finding portrait id card 
@app.post("/test_por_verify_id")
def read_root(folder:schemas.Input):
    ii=0
    folder_name=folder.folder_name
    for file in os.listdir(folder_name):
        img_path= os.path.join(folder_name,file)
        #img= cv.imread(img_path)
        rotation_angle=portrait_verify_id_model.get_rotation(img_path)
        result=rotation_angle[0]
        accuracy=float(rotation_angle[1])
        if accuracy >= 0.99:
                if result =="id":
                        shutil.move(img_path, os.path.join("/home/sid/Documents/eYRC22_TeamPhoto/test/portrait_test_id",file))   
                    #img=ndimage.rotate(img, -90,mode="nearest")
                elif result=="other_document":
                    shutil.move(img_path, os.path.join("/home/sid/Documents/eYRC22_TeamPhoto/test/portriat_test_other_doc",file))
                elif result=="others":
                     shutil.move(img_path, os.path.join("/home/sid/Documents/eYRC22_TeamPhoto/test/portrait_test_other",file))    
                print(result,accuracy)    
                #cv.imwrite(img_path,img)
        else:
            ii+=1
            print(file, f"not confirm {ii}, {accuracy} ,{result}")    
    return {"ddk":"dkk"}


# @app.get("/google_ocr")
# async def google_ocr(image_url: HttpUrl = Query(..., description="Public image URL")):
#     path= download_image_to_tempfile(image_url)
#     is_doc=doc_personal(path)
#     if is_doc!=1:
#         return {"details":"Personal photo" if is_doc==0 else "Other document" }
    
#     client = vision.ImageAnnotatorClient()

#     with open(path, "rb") as image_file:
#         content = image_file.read()

#     image = vision.Image(content=content)
#     response = client.text_detection(image=image)

#     if response.error.message:
#         raise Exception(f"{response.error.message}")

#     texts = response.text_annotations

#     if not texts:
#         return "empty string"

#     angle = get_rotation(texts)
#     print(f"Detected orientation: {angle}°")
    

#     img= cv.imread(path)
#     if angle!=0:
#         if angle==90:
#             img=ndimage.rotate(img, -90, mode="nearest")
#         else:    
#             img=ndimage.rotate(img, 360-angle, mode="nearest")
#     orientation=get_orientation(path)
#     if orientation==-1:
#         return {"detail": "don't able to get orientation"}
#     doc_type_result=-1
#     if orientation==0:
#         doc_type_result=doc_type(path)   
#     elif orientation==1:  
#         doc_type(path, is_land=False)
    
#     if doc_type_result==0:
#         print("Id")

    
    
#     # try:
#     #     image = types.Image()
#     #     image.source.image_uri = str(image_url)

#     #     response = client.text_detection(image=image)

#     #     if response.error.message:
#     #         raise HTTPException(status_code=500, detail=response.error.message)

#     #     texts = response.text_annotations
#     #     extracted_text = texts[0].description if texts else ""

#     #     return {
#     #         "image_url": image_url,
#     #         "extracted_text": extracted_text.strip()
#     #     }

#     # except Exception as e:
#     #     raise HTTPException(status_code=500, detail=f"Google OCR failed: {str(e)}")
    
# def get_rotation(texts):
#     if len(texts) < 2:
#         return 0  

#     vertices = texts[1].bounding_poly.vertices

#     if len(vertices) < 2:
#         return 0

#     v0 = vertices[0]  # Top-left
#     v1 = vertices[1]  # Top-right

#     dx = v1.x - v0.x
#     dy = v1.y - v0.y

#     angle_rad = math.atan2(dy, dx)
#     angle_deg = math.degrees(angle_rad)

#     angle_deg = angle_deg % 360

#     if angle_deg < 45 or angle_deg >= 315:
#         return 0
#     elif 45 <= angle_deg < 135:
#         return 90
#     elif 135 <= angle_deg < 225:
#         return 180
#     else:
#         return 270

    
# def download_image_to_tempfile(image_url: str) -> str:
#     response = requests.get(image_url, stream=True)
#     if response.status_code != 200:
#         raise Exception(f"Failed to download image. Status code: {response.status_code}")
#     tmp_file = NamedTemporaryFile(delete=False, suffix=".jpg")
#     with tmp_file as f:
#         shutil.copyfileobj(response.raw, f)
#     return tmp_file.name
    
# def get_orientation(file_path:str):
#         img_path= file_path
#         rotation_angle=orientation_model.get_rotation(img_path)
#         result=rotation_angle[0]
#         accuracy=float(rotation_angle[1])
#         if accuracy >= 0.99:
            
#                 if result =="landscape":
#                     return 1
#                 elif result=="portrait":
#                     return 0
#         else:
#             return -1
# def doc_personal(file_path:str):
#         img_path= file_path
#         rotation_angle=is_photo_model.get_rotation(img_path)
#         result=rotation_angle[0]
#         accuracy=float(rotation_angle[1])
#         if accuracy >= 0.5:
#                 if result =="documents":
#                     if accuracy >=0.85:
#                         return 1
#                 elif result=="personal":
#                     return 0
#         else:
#             return -1
    
# def doc_type(file_path:str, is_land=True):
#         img_path= file_path
#         doc_type=None
#         if is_land:
#             doc_type=landscape_verify_id_model.get_rotation(img_path)
#         else:
#             doc_type=portrait_verify_id_model.get_rotation(img_path)    
#         result=doc_type[0]
#         accuracy=float(doc_type[1])
#         if accuracy >= 0.99:
#                 if result =="id":
#                     return 0
#                 elif result=="other_document":
#                     return 1
#                 elif result=="others":
#                     return 2
#                 print(result,accuracy)    
#         else:
#             return -1

"""for file in os.listdir("/home/sid/Downloads/downloaded photo sample"):
    img_path=os.path.join("/home/sid/Downloads/downloaded photo sample",file)
    result=orientation_model.get_rotation(img_path)
    print(file,result)
    """
    
    
@app.get("/verif-id")
async def verif_id(request:schemas.VerifIdSchema=Depends()):
    path = download_image_to_tempfile(str(request.image_url))
    
    try:
        is_doc = doc_personal(path)
        if is_doc != 1:
            return {"details": "Personal photo" if is_doc == 0 else "Other document"}

        client = vision.ImageAnnotatorClient()

        with open(path, "rb") as image_file:
            content = image_file.read()

        image = vision.Image(content=content)
        response = client.text_detection(image=image)

        if response.error.message:
            raise HTTPException(status_code=500, detail=response.error.message)

        texts = response.text_annotations
        if not texts:
            return {"details": "empty string"}
        print(texts)
        #draw_google_ocr_boxes(path,texts)
        angle = get_rotation(texts)
        print(f"Detected orientation: {angle}°")

        rotate_and_save_image(path, angle)

        orientation = get_orientation(path)
        is_orientation_detected=True
        if orientation == -1:
            is_orientation_detected=False

        doc_type_result = doc_type(path, is_land=(orientation == 1))
        texts_result=texts[0].description.strip()
        if isLibraryCard(texts_result):
                return {"details": "Library card"}
        if isFeeReciept(texts_result):
                return {"details": "Fee Reciept"}
        name_split=request.name.split(' ')
        college_name_split=request.college_name.split(' ')
        name_weight=None
        if len(name_split)==1:
            name_weight=(1.0)
        elif len(name_split)==2:
            name_weight=(0.9,0.1)
        else:
            name_weight=(0.8,0.1,0.1)
            
                
        name_score=scoring(texts,name_split,name_weight,len(name_weight))
        college_name_score=college_score(texts, request.college_name)
        final_score=(name_score+college_name_score)/2
        department_score_value=department_score(texts,request.department_name)
        extract_year(texts_result)

        if doc_type_result == 0:
            return {
                    "doc_type": "ID Document",
                    "name_score": name_score,
                    "college_score": college_name_score,
                    "final_score":final_score,
                    "is_orientation_detected":is_orientation_detected,
                    "department_score_value":department_score_value
                    }
        elif doc_type_result == 1:
            return {
                    "doc_type": "Others Doc",
                    "name_score":name_score,
                    "college_score":college_name_score,
                    "final_score":final_score,
                    "is_orientation_detected":is_orientation_detected,
                    "department_score_value":department_score_value
                    }
        elif doc_type_result == 2:
            return {"doc_type": "Unknown Doc",
                    "name_score":name_score,
                    "college_name_score":college_name_score,
                    "final_score":final_score,
                    "is_orientation_detected":is_orientation_detected,
                    "department_score_value":department_score_value
                    }
            
        else:
            return {"details": "Low confidence in document classification"}

    finally:
        os.remove(path)



def download_image_to_tempfile(image_url: str) -> str:
    response = requests.get(image_url, stream=True)
    if response.status_code != 200:
        raise HTTPException(status_code=400, detail="Failed to download image")
    
    tmp_file = NamedTemporaryFile(delete=False, suffix=".jpg")
    with tmp_file as f:
        shutil.copyfileobj(response.raw, f)
    return tmp_file.name


def get_rotation(texts):
    if len(texts) < 2:
        return 0

    vertices = texts[1].bounding_poly.vertices
    if len(vertices) < 2:
        return 0

    dx = vertices[1].x - vertices[0].x
    dy = vertices[1].y - vertices[0].y

    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad) % 360

    if angle_deg < 45 or angle_deg >= 315:
        return 0
    elif 45 <= angle_deg < 135:
        return 90
    elif 135 <= angle_deg < 225:
        return 180
    else:
        return 270


def rotate_and_save_image(path: str, angle: int):
    img = cv.imread(path)
    if img is None:
        raise Exception("Failed to load image")

    if angle != 0:
        if angle == 90:
            rotated = ndimage.rotate(img, -90, mode="nearest")
        elif angle == 180:
            rotated = ndimage.rotate(img, 180, mode="nearest")
        elif angle == 270:
            rotated = ndimage.rotate(img, -270, mode="nearest")
        else:
            rotated = ndimage.rotate(img, 360 - angle, mode="nearest")
        cv.imwrite(path, rotated)


def get_orientation(file_path: str):
    result, accuracy = orientation_model.get_rotation(file_path)
    if float(accuracy) >= 0.95:
        return 1 if result == "landscape" else 0
    return -1


def doc_personal(file_path: str):
    result, accuracy = is_photo_model.get_rotation(file_path)
    accuracy = float(accuracy)
    if accuracy >= 0.5:
        if result == "documents" and accuracy >= 0.85:
            return 1
        elif result == "personal":
            return 0
    return -1


def doc_type(file_path: str, is_land=True):
    model = landscape_verify_id_model if is_land else portrait_verify_id_model
    result, accuracy = model.get_rotation(file_path)
    accuracy = float(accuracy)
    if accuracy >= 0.90:
        if result == "id":
            return 0
        elif result == "other_document":
            return 1
        elif result == "others":
            return 2
    return -1

def draw_google_ocr_boxes(image_path: str, text_annotations, output_path: Optional[str] = None):
    img = cv.imread(image_path)
    if img is None:
        raise Exception("Image could not be loaded.")

    for annotation in text_annotations[1:]:
        vertices = annotation.bounding_poly.vertices
        if len(vertices) != 4:
            continue 

        pts = [(v.x, v.y) for v in vertices]

        cv.polylines(img, [np.array(pts, dtype=np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
        cv.putText(img, annotation.description, (pts[0][0], pts[0][1] - 10),
           cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)


    if output_path:
        cv.imwrite(output_path, img)
        return output_path
    else:
        cv.imshow("Annotated image",img)
        cv.waitKey(0)
        return image_path


def isLibraryCard(text:str):
    count=0
    for token in libray_token:
        if token in text.lower():
            return True
    return False


def isFeeReciept(text:str):
    for token in reciept_token:
        if token in text.lower():
            print(token)
            return True    
    return False
from thefuzz import fuzz

def college_score(extacted, given):
    token_set_score = fuzz.token_set_ratio(extacted, given)
    return token_set_score    

# if weights are None then pass names as string only
def scoring(extracted,names, weights, limit):
            total_score=0
            for i,name in enumerate(names):
                token_set_score = fuzz.token_set_ratio(extracted, name)
                total_score=total_score+((token_set_score*weights[i]))
                if i==limit-1:
                    break
            return total_score
        

def department_score(extracted, given):
    token_set_score = fuzz.token_set_ratio(extracted, given)
    if token_set_score<50:
        highest_score=0
        departments=department_variants.get(given, None)
        if departments is None:
            return "Not a valid department"
        for dept in departments:
            current_score=fuzz.token_set_ratio(extracted, dept)
            highest_score=max(highest_score, current_score)
        return highest_score
                    
    else:
        return token_set_score  
    
def extract_year(extracted):
    pattern = r"\b(?:\d{1,2}[-/.]){2}(?:\d{2}|\d{4})\b|\b(?:19|20)\d{2}\b"
    matches = re.findall(pattern, extracted)
    for mat in matches:
        print(mat)
    return matches

    
        