import cv2
from cv2 import dnn_superres

sr=dnn_superres.DnnSuperResImpl_create()

path="/home/sidharth/Documents/verify_id/app/model/pretrained/EDSR_x4.pb"
sr.readModel(path)

sr.setModel('edsr',2)

sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
image = cv2.imread('/home/sidharth/Documents/rotation_data/train/0/0aDBJDO9rE.jpg')

# upsample the image
upscaled = sr.upsample(image)
# save the upscaled image
cv2.imwrite('upscaled_test.png', upscaled)