import cv2, math
import numpy as np
kernel = np.ones((5,5),np.uint8)
img = cv2.imread("/home/sidharth/Documents/verify_id/app/data/seg_train/college_id/7wUpz8yjLe.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
cv2.imshow("",mask)
cv2.waitKey(0)
dilation = cv2.dilate(mask,kernel,iterations = 1)
#cv2.imshow("", dilation)
#cv2.waitKey(0)
edges = cv2.Canny(dilation, 80, 120)
cv2.imwrite("/home/sidharth/Documents/verify_id/app/data/seg_train/7wUpz8yjLe.jpg", edges)

