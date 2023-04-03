import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets


def get_dataloaders(path = './data/affectnet',bs = 32):
    test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.TenCrop(200),
    transforms.Lambda(lambda crops: torch.stack(
        [transforms.ToTensor()(crop) for crop in crops])),
    transforms.Lambda(lambda tensors: torch.stack(
        [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(t) for t in tensors])),
    ])
    train_transform = transforms.Compose([
         transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.2)),
        transforms.RandomApply([transforms.ColorJitter(
            brightness=0.5, contrast=0.5, saturation=0.5)], p=0.5),
        transforms.RandomApply(
            [transforms.RandomAffine(0, translate=(0.2, 0.2))], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.RandomRotation(10)], p=0.5),
        transforms.FiveCrop(200),
        transforms.Lambda(lambda crops: torch.stack(
            [transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda tensors: torch.stack(
            [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(t) for t in tensors])),
        transforms.Lambda(lambda tensors: torch.stack(
            [transforms.RandomErasing()(t) for t in tensors])),
    ])
#     train_transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomApply([
#                 transforms.RandomAffine(20, scale=(0.8, 1), translate=(0.2, 0.2)),
#             ], p=0.7),
#         transforms.RandomApply([transforms.RandomRotation(20)], p=0.5),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225]),
#         transforms.RandomErasing(),
#         ])
#     test_transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])])
    train_dataset = datasets.ImageFolder(root=os.path.join(path, 'train'), transform=train_transform)
    test_dataset = datasets.ImageFolder(root=os.path.join(path, 'val_class'), transform=test_transform)
      
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=bs, shuffle=False)
    
    return train_loader,test_loader
 
        
    
    
    
    