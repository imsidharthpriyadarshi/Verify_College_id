import easyocr
import numpy as np
import os
import cv2
import shutil

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

def infer_orientation_easyocr(image_path):
    reader = easyocr.Reader(['en'])  # Specify languages as needed
    result = reader.readtext(image_path, detail=1, paragraph=False)
    
    if not result:
        return None, "No Text Detected"
    #print(result)
    # Analyze text orientations
    angles = []
    for (bbox, text, confidence) in result:
        # Calculate angle based on bounding box
        # Example: Compute the angle between the first two points
        (x0, y0), (x1, y1), _, _ = bbox
        angle = np.arctan2(y1 - y0, x1 - x0) * (180.0 / np.pi)
        angles.append(angle)
    
    # Average angle
    if angles:
        avg_angle = np.mean(angles)
        # Round to nearest 90 degrees
        rotation_angle = (round(avg_angle / 90) * 90) % 360
        return rotation_angle, "EasyOCR Detection"
    else:
        return None, "No Angles Detected"

# Example Usage
for file in os.listdir("/home/sidharth/Documents/eYRC22_TeamPhoto/teamPhoto"):
    file_path= os.path.join("/home/sidharth/Documents/eYRC22_TeamPhoto/teamPhoto",file)
    rotation_angle, orientation = infer_orientation_easyocr(file_path)
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
          
