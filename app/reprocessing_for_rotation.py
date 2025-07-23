import cv2
import matplotlib.pyplot as plt

def apply_edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)
    return edges

from skimage.feature import hog
from skimage.color import rgb2gray

def extract_hog_features(image):
    gray = rgb2gray(image)
    features, hog_image = hog(gray, orientations=8, pixels_per_cell=(16, 16),
                              cells_per_block=(1, 1), visualize=True)
    return hog_image

fig = plt.figure(figsize=(10, 7)) 
rows =3
columns=2
img_name="0aDBJDO9rE.jpg"
img_name90="0aDBJDO9rE.jpg"
img_name180="0aDBJDO9rE.jpg"
img_name270="0aDBJDO9rE.jpg"

fig.add_subplot(rows, columns, 1)
# showing image 
plt.imshow(cv2.imread(f"/home/sidharth/Documents/rotation_data/train/0/{img_name}"))
plt.axis('off')
plt.title("First")
# Adds a subplot at the 2nd position 
fig.add_subplot(rows, columns, 2) 
  
# showing image 
plt.imshow(extract_hog_features(cv2.imread(f"/home/sidharth/Documents/rotation_data/train/0/{img_name}"))) 
plt.axis('off') 
plt.title("Second") 
  
#Adds a subplot at the 3rd position 
fig.add_subplot(rows, columns, 3) 
# showing image 
plt.imshow(extract_hog_features(cv2.imread(f"/home/sidharth/Documents/rotation_data/train/90/{img_name}"))) 
plt.axis('off') 
plt.title("Third") 


fig.add_subplot(rows, columns, 4)
                    
plt.imshow(extract_hog_features(cv2.imread(f"/home/sidharth/Documents/rotation_data/train/180/{img_name}")))
plt.axis("off")
plt.title("Fourth")


fig.add_subplot(rows, columns, 5)

plt.imshow(extract_hog_features(cv2.imread(f"/home/sidharth/Documents/rotation_data/train/270/{img_name}")))
plt.axis("off")
plt.title("Fifth")

plt.show()
