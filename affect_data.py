import os
import random
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset, Subset
from PIL import Image
class_mapping = {
    'class001': 0,
    'class002': 1,
    'class003': 2,
    'class004': 3,
    'class005': 4,
    'class006': 5,
    'class007': 6,
    'class008': 7
}
class AffectNetDataset(Dataset):
    def __init__(self, data_dir, transform=None, target_class="class001", class_mapping=None, mode='train'):
        self.data_dir = data_dir
        self.transform = transform
        self.target_class = target_class
        self.class_mapping = class_mapping
        self.mode = mode
        self.data = []
        self.labels = []

        if self.mode == 'train':
            emotion_dir = os.path.join(data_dir, 'train', target_class)
            for img_name in os.listdir(emotion_dir):
                img_path = os.path.join(emotion_dir, img_name)
                self.data.append(img_path)
                self.labels.append(self.class_mapping[target_class])

        elif self.mode == 'val_class':
            pos_samples = []
            neg_samples = []
            for emotion, label in self.class_mapping.items():
                emotion_dir = os.path.join(data_dir, 'val_class', emotion)
                for img_name in os.listdir(emotion_dir):
                    img_path = os.path.join(emotion_dir, img_name)
                    if emotion == target_class:
                        pos_samples.append((img_path, 1))
                    else:
                        neg_samples.append((img_path, -1))

            random.shuffle(neg_samples)
            balanced_samples = pos_samples + neg_samples[:len(pos_samples)]
            random.shuffle(balanced_samples)
            self.data, self.labels = zip(*balanced_samples)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path = self.data[index]
        label = self.labels[index]
        img = Image.open(img_path).convert("RGB")
        
        if self.transform:
            img = self.transform(img)

        return img, label


def get_dataloaders(path,bs,target_class):
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
    

    train_dataset = AffectNetDataset(path, transform=train_transform, target_class=target_class, class_mapping=class_mapping, mode='train')
    test_dataset = AffectNetDataset(path, transform=test_transform, target_class=target_class, class_mapping=class_mapping, mode='val_class')


    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=1)
    
    return train_loader,test_loader