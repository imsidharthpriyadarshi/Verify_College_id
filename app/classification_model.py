import torch
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import warnings

from torch import nn, optim

warnings.simplefilter("ignore")


train_transform =transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomRotation(7.5),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])




])

test_transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])


])

train_dir='../archive/seg_train/seg_train'
test_dir='../archive/seg_test/seg_test'
bs=50

train_data=datasets.ImageFolder(root=train_dir, transform=train_transform)
test_data=datasets.ImageFolder(root=test_dir, transform=test_transform)

print(len(train_data), len(test_data))

train_data_loader=DataLoader(train_data,batch_size=bs, shuffle=True)
test_data_loader=DataLoader(test_data, batch_size=bs, shuffle=False)



train_data_loader.dataset.classes

resnet50=models.resnet50(pretrained=True)

print(resnet50)


for param in resnet50.parameters():
    param.requires_grad=True

resnet50.fc=nn.Sequential(
    nn.Linear(2048,256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256,128),
    nn.ReLU(),
    nn.Linear(128,6),
    nn.LogSoftmax(dim=1)
)


device= 'cpu'

if torch.cuda.is_available():
    device='cuda'
model=resnet50
print(device)    


model=model.to(device)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())



epochs = 50
training_loss = []
testing_loss = []
accuracy_list = []
for epoch in range(epochs):
    print("Epoch: {}/{}".format(epoch+1,epochs))
    model.train()

    train_loss = 0.0
    test_loss = 0.0
    test_acc = 0.0



    for i , (inputs,labels) in enumerate(train_data_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        output = model(inputs)
        loss = loss_func(output,labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()*inputs.size(0)
#         writer.add_scalar('loss', loss, epoch)
        print("Batch number: {:03d}, Training_Loss: {:.4f},".format(i, loss.item()))


    with torch.no_grad():
        model.eval()
        num_correct = 0
        num_examples = 0


        for j , (inputs, target) in enumerate(test_data_loader):
            inputs = inputs.to(device)
            target = target.to(device)

            output = model(inputs)
            loss = loss_func(output, target)
            correct = torch.eq(torch.max(torch.functional.F.softmax(output), dim=1)[1], target).view(-1)
            num_correct += torch.sum(correct).item()
            num_examples += correct.shape[0]
            test_loss += loss.item()*inputs.size(0)
            print("Test Batch number: {:03d}, Test_Loss: {:.4f}, Accuracy:{:.2f}".format(j, loss.item(),num_correct/num_examples))
#             writer.add_scalar('accuracy', num_correct / num_examples, epoch)
#             writer.add_scalar('Test_loss', loss, epoch)
#             test_acc += num_correct





    avg_train_loss = train_loss/len(train_data)
    avg_test_loss = test_loss/len(test_data)
    avg_acc = num_correct/len(test_data)
    training_loss.append(avg_train_loss)
    testing_loss.append(avg_test_loss)
    accuracy_list.append(avg_acc)
    print(avg_acc, avg_train_loss)
    if avg_train_loss<0.12 and avg_acc >.85:
        torch.save(model,'./iterdrop'+str(epoch)+'.pth')
#     if avg_test_loss == avg_train_loss:
#         break

