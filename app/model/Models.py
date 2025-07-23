import torch
from torchvision import transforms
import matplotlib.pyplot as plt
# from torch.utils.tensorboard import SummaryWriter
import warnings
from PIL import Image
import os, shutil
from .models import rotnet_model

warnings.simplefilter("ignore")

class Model():
    def __init__(self,model_path,classes, test_transform):
        self.test_transform = test_transform
        #self.model=rotnet_model.RotNet()
        #self.model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
        self.model = torch.load(model_path, map_location=torch.device('cpu'))
        self.classes = classes#['0', '180', '270', '90']
        # classes = ['documents', 'personal']
    
    
    def get_rotation(self,image):
        result=self.predict(image)
        return result
        
    def predict(self, test_image_name):

        transform = self.test_transform
        test_image=None
        try:
            test_image = Image.open(test_image_name).convert('RGB')
        #plt.imshow(test_image)
        except Exception as e:
            print(test_image_name)
            return


        test_image_tensor = transform(test_image)
    #     print(test_image_tensor.size())

        if torch.cuda.is_available():
            test_image_tensor = test_image_tensor.view(1, 3, 224, 224).cuda()
        else:
            test_image_tensor = test_image_tensor.view(1, 3, 224, 224)

        with torch.no_grad():
            self.model.eval()
            # Model outputs log probabilities
            out = self.model(test_image_tensor)
            _ , b = (torch.max(out,1))
    #         print(out)
    #         print(train_data_loader.dataset.classes[b.item()])
            ps = torch.exp(out)
            topk, topclass = ps.topk(1, dim=1)
    #     print(ps)
    #     print(topk,topclass)
        return self.classes[b.item()], topk.cpu().numpy()[0][0]