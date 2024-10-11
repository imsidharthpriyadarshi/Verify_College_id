import torch
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import numpy as np
import warnings
import matplotlib.pyplot as plt
#warnings.simplefilter("ignore")
print(torch.__version__)

#print(os.getcwd())
from torch import nn, optim
from PIL import Image
test_transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])


])



model = torch.load('/home/sidharth/Documents/verify_id/app/iterdrop48.pth', map_location=torch.device('cpu'))

classes = ['aadhar', 'college_id', 'pancard', 'personal_photo']
def predict(model, test_image_name):
    test_image = Image.open(test_image_name)
    plt.imshow(test_image)

    test_image_tensor = test_transform(test_image).unsqueeze(0)  # Add batch dimension

    if torch.cuda.is_available():
        model = model.cuda()
        test_image_tensor = test_image_tensor.cuda()

    model.eval()  # Ensure model is in evaluation mode

    with torch.no_grad():  # Disable gradient computation for inference
        out = model(test_image_tensor)  # Get raw outputs (logits or probabilities)

        _, predicted_class = torch.max(out, 1)  # Predicted class index
        top_class_prob = torch.softmax(out, dim=1)[0, predicted_class].item()  # Predicted class probability

    return classes[predicted_class.item()], top_class_prob


predict(model,"/home/sidharth/Documents/verify_id/app/model_land_right/seg_test/college_id/xDZh7KwUzu.jpg")