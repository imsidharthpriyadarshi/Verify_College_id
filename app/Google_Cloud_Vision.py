"""
need to install google-cloud-vision (GCP SDK) from conda -c conda-forge
conda install -c conda-forge pillow=10.1.0 pandas=2.1.2 google-cloud-vision=3.4.5 scikit-learn=1.3.2 ipykernel jupyterlab notebook python=3.12.0
to set up in jupyterlabs:
python -m ipykernel install --user --name=gcp-cloud-vision
repo: https://github.com/donaldsrepo/gcp-solution
"""

import os
from os import listdir
from os.path import isfile, join
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] ='app/visions_key.json'
from google.cloud import vision
import time

from google.cloud import vision
import math

def detect_text(path):
    client = vision.ImageAnnotatorClient()

    with open(path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client.text_detection(image=image)

    if response.error.message:
        raise Exception(f"{response.error.message}")

    texts = response.text_annotations

    if not texts:
        return "empty string"

    orientation = get_orientation(texts)
    print(f"Detected orientation: {orientation}°")

    return texts[0].description.strip()


def get_orientation(texts):
    if len(texts) < 2:
        return 0  

    vertices = texts[1].bounding_poly.vertices

    if len(vertices) < 2:
        return 0

    v0 = vertices[0]  # Top-left
    v1 = vertices[1]  # Top-right

    dx = v1.x - v0.x
    dy = v1.y - v0.y

    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)

    angle_deg = angle_deg % 360

    if angle_deg < 45 or angle_deg >= 315:
        return 0
    elif 45 <= angle_deg < 135:
        return 90
    elif 135 <= angle_deg < 225:
        return 180
    else:
        return 270


def main():
    mypath = "/home/sid/Documents/datasets/doubt/QrUAKgxoPa.jpg"
    text = detect_text(mypath)
        
    print(mypath)
    print(text)

if __name__ == "__main__":
     main()
