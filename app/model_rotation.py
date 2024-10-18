import torch
from torchvision import models, datasets,transforms
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
import torch.nn.functional as F

train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
train_dir = '/home/sidharth/Verify_College_id/app/rotation_data/seg_train'
test_dir = '/home/sidharth/Verify_College_id/app/rotation_data/test'

bs = 100
train_data = datasets.ImageFolder(root = train_dir,transform=train_transform)
test_data = datasets.ImageFolder(root=test_dir, transform=test_transform)

print(len(train_data),len(test_data))


train_data_loader = DataLoader(train_data,batch_size=bs,shuffle=True)
test_data_loader = DataLoader(test_data,batch_size=bs, shuffle=False)


train_data_loader.dataset.classes
torch.cuda.empty_cache()

resnet50 = models.resnet50(pretrained=True)

for param in resnet50.parameters():
    param.require_grad = True

resnet50.fc = nn.Sequential(
    nn.Linear(2048,256),
    nn.ReLU(),
    nn.Dropout(0.6),
    nn.Linear(256, 4),
    nn.LogSoftmax(dim=1)    

)    

device = 'cuda'
model = resnet50

model = model.to(device)
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
        if input ==None or labels ==None:
            continue
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
            if inputs==None or target==None:
                continue
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
    if avg_train_loss<0.12 and avg_acc >.95:
        try:
            torch.save(model,'./iterdrop'+str(epoch)+'.pth')
            print("saved ->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
#     if avg_test_loss == avg_train_loss:
#         break
        except Exception as e:
            print(e)
    

     