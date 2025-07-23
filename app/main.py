import os

from fastapi import FastAPI
from app.schemas import schemas
import cv2 as cv
from app.model import Models,Modelss
from app.preprocessing import transformations
from scipy import ndimage

from app.ocr_operation import ocr_operations
import shutil
app= FastAPI()


rotation_model= Modelss.Modelss("/home/sidharth/Documents/verify_id/rotnet_final1.pth", ['0', '180', '270', '90'],transformations.rotation_test_transform)
is_photo_model=Models.Model("app/model/trained/document_personal_classification_iterdrop49.pth", ['documents', 'personal'],transformations.classification_test_transform)
orientation_model=Models.Model("app/model/trained/orientation_classification_iterdrop49.pth",['landscape','portrait'],transformations.classification_test_transform)
landscape_verify_id_model=Models.Model("/home/sidharth/Documents/verify_id/app/model/trained/land_id_classification_iterdrop49.pth",['id','other_document','others'],transformations.classification_test_transform)
portrait_verify_id_model=Models.Model("/home/sidharth/Documents/verify_id/app/model/trained/por_id_classification_iterdrop49.pth",['id','other_document','others'],transformations.classification_test_transform)

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
                    shutil.move(img_path, os.path.join("/home/sidharth/Documents/eYRC22_TeamPhoto/90",file))
                    print(90, "moved")
                    #img=ndimage.rotate(img, -90,mode="nearest")
                elif angle==180:
                    # if accuracy !=1:
                    #     ii+=1
                    #     print(180, "continue")
                    #     continue
                    shutil.move(img_path, os.path.join("/home/sidharth/Documents/eYRC22_TeamPhoto/180",file))
                    print(180, "moved")
                elif angle==270:
                    # if accuracy !=1:
                    #     ii+=1
                    #     print(270, "continue")
                    #     continue
                    shutil.move(img_path, os.path.join("/home/sidharth/Documents/eYRC22_TeamPhoto/270",file))
                    print(270, "moved")
                elif angle ==0:
                    shutil.move(img_path, os.path.join("/home/sidharth/Documents/eYRC22_TeamPhoto/0",file))
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
                        shutil.move(img_path, os.path.join("/home/sidharth/Documents/eYRC22_TeamPhoto/00_doc",file))
                    else:
                        ii+=1    
                    #img=ndimage.rotate(img, -90,mode="nearest")
                elif result=="personal":
                    shutil.move(img_path, os.path.join("/home/sidharth/Documents/eYRC22_TeamPhoto/00_personal",file))
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
                        shutil.move(img_path, os.path.join("/home/sidharth/Documents/eYRC22_TeamPhoto/test_landscape",file))   
                    #img=ndimage.rotate(img, -90,mode="nearest")
                elif result=="portrait":
                    shutil.move(img_path, os.path.join("/home/sidharth/Documents/eYRC22_TeamPhoto/test_portrait",file))
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
                        shutil.move(img_path, os.path.join("/home/sidharth/Documents/eYRC22_TeamPhoto/test_college_id",file))   
                    #img=ndimage.rotate(img, -90,mode="nearest")
                elif result=="other_document":
                    shutil.move(img_path, os.path.join("/home/sidharth/Documents/eYRC22_TeamPhoto/test_other_doc",file))
                elif result=="others":
                     shutil.move(img_path, os.path.join("/home/sidharth/Documents/eYRC22_TeamPhoto/test_other",file))    
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
                        shutil.move(img_path, os.path.join("/home/sidharth/Documents/eYRC22_TeamPhoto/test/portrait_test_id",file))   
                    #img=ndimage.rotate(img, -90,mode="nearest")
                elif result=="other_document":
                    shutil.move(img_path, os.path.join("/home/sidharth/Documents/eYRC22_TeamPhoto/test/portriat_test_other_doc",file))
                elif result=="others":
                     shutil.move(img_path, os.path.join("/home/sidharth/Documents/eYRC22_TeamPhoto/test/portrait_test_other",file))    
                print(result,accuracy)    
                #cv.imwrite(img_path,img)
        else:
            ii+=1
            print(file, f"not confirm {ii}, {accuracy} ,{result}")    
    return {"ddk":"dkk"}
#res= orientation_model.get_rotation("/home/sidharth/Documents/datasets/doubt/untKyAq1lr.jpg")
#print(res)
"""for file in os.listdir("/home/sidharth/Downloads/downloaded photo sample"):
    img_path=os.path.join("/home/sidharth/Downloads/downloaded photo sample",file)
    result=orientation_model.get_rotation(img_path)
    print(file,result)
    """
#Next day 