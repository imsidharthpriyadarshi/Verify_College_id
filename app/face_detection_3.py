"""import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision



BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a face detector instance with the image mode:
options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path=''),
    running_mode=VisionRunningMode.IMAGE)
with FaceDetector.create_from_options(options) as detector:
    mp_image = mp.Image.create_from_file('')
    face_detector_result = detector.detect(mp_image)
    print(face_detector_result)


"""


# STEP 1: Import the necessary modules.
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os 
import cv2
import time
path="/home/sidharth/Documents/verify_id/app/data/seg_train/potrait"

# STEP 2: Create an FaceDetector object.
base_options = python.BaseOptions(model_asset_path='/home/sidharth/Documents/verify_id/app/blaze_face_short_range.tflite')
options = vision.FaceDetectorOptions(base_options=base_options)
detector = vision.FaceDetector.create_from_options(options)
for file in os.listdir(path):
  
# STEP 3: Load the input image.
    image = mp.Image.create_from_file(os.path.join(path, file))
    img=cv2.imread(os.path.join(path, file))

    # STEP 4: Detect faces in the input image.
    detection_result = detector.detect(image)

    # STEP 5: Process the detection result. In this case, visualize it.
    for detection in detection_result.detections:
        # Draw bounding_box
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(img, start_point, end_point, (0,0,255), 3)
    cv2.namedWindow("Resized_Window",cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Resized_Window", 300, 700) 
    cv2.imshow("Resized_Window", img)
    cv2.waitKey(3)
    time.sleep(3)