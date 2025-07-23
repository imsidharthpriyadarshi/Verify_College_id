import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import warnings

warnings.simplefilter("ignore")

# Define RotNet model
class RotNet(nn.Module):
    def __init__(self, num_classes=4):
        super(RotNet, self).__init__()
        self.base_model = models.resnet50(pretrained=True)
        for param in self.base_model.parameters():
            param.requires_grad = False  # Freeze ResNet layers

        # Replace the final layer
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
            )

    def forward(self, x):
        return self.base_model(x)

# Prepare dataset
mean = [0.6416891813278198, 0.6349133849143982, 0.645208477973938]
std_dev = [0.2710784673690796, 0.26727473735809326, 0.2692960500717163]

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std_dev)
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std_dev)
])

train_dir = '/home/sidharth/Documents/rotation_data/train'
test_dir = '/home/sidharth/Documents/rotation_data/test'
bs = 124

train_data = datasets.ImageFolder(root=train_dir, transform=train_transform)
test_data = datasets.ImageFolder(root=test_dir, transform=test_transform)
train_data_loader = DataLoader(train_data, batch_size=bs, shuffle=True)
test_data_loader = DataLoader(test_data, batch_size=bs, shuffle=False)

# Initialize model, loss function, and optimizer
model = RotNet(num_classes=4).to('cuda' if torch.cuda.is_available() else 'cpu')
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model = torch.load("/home/sidharth/Documents/verify_id/app/model/trained/rotnet_final.pth", map_location=torch.device('cpu'))
# model.eval()
# torch.save(model.state_dict(), "./rotnet_final1.pth")
# # Training loop
# epochs = 50
# training_loss = []
# testing_loss = []
# accuracy_list = []

# for epoch in range(epochs):
#     print(f"Epoch: {epoch + 1}/{epochs}")
#     model.train()

#     train_loss = 0.0
#     for i, (inputs, labels) in enumerate(train_data_loader):
#         inputs, labels = inputs.to(device), labels.to(device)

#         optimizer.zero_grad()
#         output = model(inputs)
#         loss = loss_func(output, labels)
#         loss.backward()
#         optimizer.step()

#         train_loss += loss.item() * inputs.size(0)
#         print(f"Batch number: {i:03d}, Training Loss: {loss.item():.4f}")

#     with torch.no_grad():
#         model.eval()
#         test_loss = 0.0
#         num_correct = 0
#         num_examples = 0

#         for j, (inputs, target) in enumerate(test_data_loader):
#             inputs, target = inputs.to(device), target.to(device)

#             output = model(inputs)
#             loss = loss_func(output, target)
#             correct = torch.eq(torch.max(output, dim=1)[1], target).view(-1)
#             num_correct += torch.sum(correct).item()
#             num_examples += correct.shape[0]
#             test_loss += loss.item() * inputs.size(0)
#             print(f"Test Batch number: {j:03d}, Test Loss: {loss.item():.4f}, Accuracy: {num_correct / num_examples:.2f}")

#     avg_train_loss = train_loss / len(train_data)
#     avg_test_loss = test_loss / len(test_data)
#     avg_acc = num_correct / len(test_data)

#     training_loss.append(avg_train_loss)
#     testing_loss.append(avg_test_loss)
#     accuracy_list.append(avg_acc)

#     print(f"Validation Accuracy: {avg_acc:.4f}, Training Loss: {avg_train_loss:.4f}")

#     if avg_train_loss < 0.12 and avg_acc > 0.85:
#         torch.save(model, f'./rotnet_model_epoch_{epoch}.pth')
#         print("Model saved!")

# min_training_loss = min(training_loss)
# min_epoch = training_loss.index(min_training_loss)
# torch.save(model, "./rotnet_final.pth")
