from pydantic import BaseModel,HttpUrl



class Input(BaseModel):
    folder_name:str
    
class FinalInput(BaseModel):
    url:str

class VerifIdSchema(BaseModel):
    image_url:HttpUrl
    name:str
    college_name:str
    department_name:str
    
         