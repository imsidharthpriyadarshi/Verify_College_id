import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def calculate_mean_and_std(dataset_dir, batch_size=64):

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])
    

    dataset = datasets.ImageFolder(root=dataset_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_pixels = 0
    
    print("Calculating mean and standard deviation...")
    
    # Iterate through the dataset to calculate mean
    for images, _ in dataloader:
        # Images are in [batch, channels, height, width] format
        print("working>>")
        batch_pixels = images.size(0) * images.size(2) * images.size(3)
        total_pixels += batch_pixels
        mean += images.sum(dim=[0, 2, 3])  # Sum over batch, height, and width
    
    mean /= total_pixels  # Normalize by total pixels
    
    # Iterate through the dataset again to calculate standard deviation
    for images, _ in dataloader:
        std += ((images - mean.view(1, 3, 1, 1)) ** 2).sum(dim=[0, 2, 3])
    
    std = torch.sqrt(std / total_pixels)
    
    return mean, std

dataset_directory = "/home/sidharth/Documents/rotation_data/train"

mean, std = calculate_mean_and_std(dataset_directory)
print("Mean:", mean.tolist())
print("Standard Deviation:", std.tolist())
