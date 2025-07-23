from pytesseract import pytesseract, Output
import cv2
import os, shutil

# Path to the Tesseract executable
pytesseract.tesseract_cmd = '/bin/tesseract'

def preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to binarize the image
    adaptive_thresh = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2
)

    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Denoise the image
    denoised = cv2.medianBlur(adaptive_thresh, 3)
    
    return denoised

def infer_orientation(image_path):
    # Load the image
    # image = cv2.imread(image_path)
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray=preprocess_image(image_path)

    # Extract orientation and script detection details
    osd_data = pytesseract.image_to_osd(gray, output_type=Output.DICT)
    
    # Extract rotation angle
    rotation_angle = osd_data.get("rotate", 0)
    orientation = osd_data.get("orientation", None)
    
    return rotation_angle, orientation

# Example Usage
# image_path = "/home/sidharth/Documents/rotation_data/train/270/0aDBJDO9rE.jpg"
# rotation_angle, orientation = infer_orientation(image_path)

# print(f"Rotation Angle: {rotation_angle}°")
# print(f"Orientation: {orientation}")
for file in os.listdir("/home/sidharth/Documents/eYRC22_TeamPhoto/teamPhoto"):
    file_path= os.path.join("/home/sidharth/Documents/eYRC22_TeamPhoto/teamPhoto",file)
    rotation_angle, orientation = infer_orientation(file_path)
    print(f"Detected Rotation Angle: {rotation_angle}° {orientation}")
    if rotation_angle !=None:
        if rotation_angle==0:
            shutil.move(file_path, os.path.join("/home/sidharth/Documents/eYRC22_TeamPhoto/0",file))
        elif rotation_angle==90:
            shutil.move(file_path, os.path.join("/home/sidharth/Documents/eYRC22_TeamPhoto/90",file))
        elif rotation_angle==180:
            shutil.move(file_path, os.path.join("/home/sidharth/Documents/eYRC22_TeamPhoto/180",file))
        elif rotation_angle==270:  
            shutil.move(file_path, os.path.join("/home/sidharth/Documents/eYRC22_TeamPhoto/270",file))
          