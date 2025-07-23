from torchvision import transforms

mean = [0.6416891813278198, 0.6349133849143982, 0.645208477973938]
std_dev = [0.2710784673690796, 0.26727473735809326, 0.2692960500717163]

rotation_test_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,
                            std=std_dev)
    ])
classification_test_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])



# Mean and std_dev for rotation_dataset for 64 batch size
# Mean: [0.6416890621185303, 0.6349132061004639, 0.6452087759971619]
# Standard Deviation: [0.2710784375667572, 0.26727473735809326, 0.2692961096763611]