
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


transform_validation = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

training_set = torchvision.datasets.ImageFolder(
    root='/home/ayb/Documents/datasets/ILSVRC/Data/CLS-LOC/train/', transform=transform_train
)
validation_set = torchvision.datasets.ImageFolder(
    root='/home/ayb/Documents/datasets/ILSVRC/Data/CLS-LOC/val/', transform=transform_validation
)


training_loader = torch.utils.data.DataLoader(
    training_set, batch_size=32, shuffle=True, num_workers=6, pin_memory=True
)

validation_loader = torch.utils.data.DataLoader(
    validation_set, batch_size=65, shuffle=False, num_workers=6, pin_memory=True
)
