import pandas as pd
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from imblearn.under_sampling import RandomUnderSampler

class Fer2013Dataset(Dataset):
    def __init__(self, csv_file, target_class,transform=None, mode='train'):
        self.transform = transform
        self.mode = mode
        
        # 读取csv文件
        data = pd.read_csv(csv_file)

        # 划分数据集
        if mode == 'train':
            data = data[data['emotion'] == target_class] # 只选择anger类
            self.data = data[data['Usage'] == 'Training']
        elif mode == 'test':
            self.data = data[data['Usage'] == 'PublicTest']
        elif mode == 'val':
            self.data = data[data['Usage'] == 'PrivateTest']
            
        if(mode != 'train'):
            for i in range(len(self.data)):
                if self.data.iloc[i, 0] == target_class:
                    self.data.iloc[i, 0] = 1
                else:
                    self.data.iloc[i, 0] = -1
            X = self.data.iloc[:, 1:].values
            y = self.data.iloc[:, 0].values
            rus = RandomUnderSampler(sampling_strategy='auto', replacement=False,random_state=None)
            self.X_resampled, self.y_resampled = rus.fit_resample(X,y)
            self.data = pd.DataFrame(np.concatenate([self.y_resampled.reshape(-1, 1), self.X_resampled], axis=1),
                                     columns=self.data.columns)
        
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = np.asarray([int(x) for x in self.data.iloc[idx, 1].split()]).reshape(48, 48).astype(np.uint8)
#         if img_data.size == 0:
#             raise ValueError(f"Empty image data for index {idx}")
#         img = img_data.reshape(48, 48)
        img = Image.fromarray(img)
        label = torch.tensor(self.data.iloc[idx, 0]).type(torch.long)

            

        if self.transform is not None:
            img = self.transform(img)

        return img, label
    
def get_dataloaders(path, bs, target_class):
    # 定义预处理和数据增强的操作
    mu, st = 0, 255
    test_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.TenCrop(40),
        transforms.Lambda(lambda crops: torch.stack(
            [transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda tensors: torch.stack(
            [transforms.Normalize(mean=(mu,), std=(st,))(t) for t in tensors])),
    ])

    train_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.RandomResizedCrop(48, scale=(0.8, 1.2)),
        transforms.RandomApply([transforms.ColorJitter(
            brightness=0.5, contrast=0.5, saturation=0.5)], p=0.5),
        transforms.RandomApply(
            [transforms.RandomAffine(0, translate=(0.2, 0.2))], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.RandomRotation(10)], p=0.5),
        transforms.FiveCrop(40),
        transforms.Lambda(lambda crops: torch.stack(
            [transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda tensors: torch.stack(
            [transforms.Normalize(mean=(mu,), std=(st,))(t) for t in tensors])),
        transforms.Lambda(lambda tensors: torch.stack(
            [transforms.RandomErasing()(t) for t in tensors])),
    ])

    # 加载数据集
    train_dataset = Fer2013Dataset(path, target_class,transform=train_transform, mode='train')
    test_dataset = Fer2013Dataset(path, target_class,transform=test_transform, mode='test')
    val_dataset = Fer2013Dataset(path, target_class,transform=test_transform, mode='val')

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=1 )
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False, num_workers=1)

    # 打印数据集和加载器的信息
#     print(f"Train loader size: {len(train_dataset)}")
#     print(f"Test loader size: {len(test_dataset)}")
#     print(f"Validation loader size: {len(val_dataset)}")
#     print(f"Train loader size: {len(train_loader)}")
#     print(f"Test loader size: {len(test_loader)}")
#     print(f"Validation loader size: {len(val_loader)}")
    
    return train_loader,test_loader,val_loader
