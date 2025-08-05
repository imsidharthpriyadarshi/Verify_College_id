import pytesseract as tess
from thefuzz import fuzz
import cv2 as cv
from matplotlib import pyplot as plt 
import numpy as np
fig = plt.figure(figsize=(10, 7)) 
def score(extracted, actual, field_type):
    levenshtein_score = fuzz.ratio(extracted, actual)
    token_sort_score = fuzz.token_sort_ratio(extracted, actual)
    token_set_score = fuzz.token_set_ratio(extracted, actual)
    print(levenshtein_score, token_sort_score, token_set_score)
    weights = {
        'name': (0.05, 0.05, 0.9),
        'college': (0.05, 0.05, 0.9),
    }

    w1, w2, w3 = weights.get(field_type, (0.3, 0.4, 0.3)) 

    combined_score = (levenshtein_score * w1 +
                      token_sort_score * w2 +
                      token_set_score * w3)

    return combined_score

img="/home/sid/Documents/rotation_data/test/0/upwa5LnJz8.jpg"
img=cv.imread(img)
img=cv.cvtColor(img, cv.COLOR_BGR2RGB)
gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
kernel = np.array([[-1,-1,-1,-1,-1],[-1,-1,-1,-1,-1], [-1,-1,25,-1,-1], [-1,-1,-1,-1,-1],[-1,-1,-1,-1,-1]])
sharp_img = cv.filter2D(gray, -1, kernel)
kernel=np.ones((3,3), np.uint8)
thresh= cv.threshold(gray,110,255,cv.THRESH_BINARY)[1]
opening = cv.morphologyEx(sharp_img, cv.MORPH_OPEN, kernel)
median = cv.GaussianBlur(thresh,(3,3),0)
text=tess.image_to_string(gray)
rows =2
columns=2
fig.add_subplot(rows, columns, 1) 
  
# showing image 
plt.imshow(img) 
plt.axis('off') 
plt.title("First")

fig.add_subplot(rows, columns, 2) 
  
# showing image 
plt.imshow(gray) 
plt.axis('off') 
plt.title("Second") 
  
# Adds a subplot at the 3rd position 
fig.add_subplot(rows, columns, 3) 
  
# showing image 
plt.imshow(thresh) 
plt.axis('off') 
plt.title("Third") 



fig.add_subplot(rows, columns, 4)

plt.imshow(opening)
plt.axis("off")
plt.title("Fourth")

plt.show()
print(text)

print(score(text.lower(), "davari vaishnavi ",'name'))
print(score(text.lower(), "vishwakarma institute of technology", 'college'))



