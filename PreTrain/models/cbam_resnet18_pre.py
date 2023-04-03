import torch
import torch.nn as nn
import torch
import torch.nn as nn
from torchvision.models import resnet18


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // reduction_ratio, kernel_size=1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // reduction_ratio, in_planes, kernel_size=1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_planes, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x
    
class ResNet18_CBAM(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet18_CBAM, self).__init__()
        self.resnet = resnet18(pretrained=pretrained)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 8)
        
        # 添加CBAM模块
        self.cbam = CBAM(64)
        self.cbam1 = CBAM(128)
        self.cbam2 = CBAM(256)
        self.cbam3 = CBAM(512)

    def _make_layer(self, block, planes, blocks, stride=1):
        layers = []
        for i in range(blocks):
            if i == 0:
                layers.append(block(self.inplanes, planes, stride))
            else:
                layers.append(block(self.inplanes, planes))

            # 添加CBAM
            if planes == 64:
                layers.append(self.cbam)
            elif planes == 128:
                layers.append(self.cbam1)
            elif planes == 256:
                layers.append(self.cbam2)
            elif planes == 512:
                layers.append(self.cbam3)

        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.resnet.fc(x)

        return x

def CBAM_ResNet18():
    return ResNet18_CBAM(pretrained=True)