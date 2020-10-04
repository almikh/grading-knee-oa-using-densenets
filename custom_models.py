
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)
    
class AntonyCnn(nn.Module):
    def __init__(self, num_classes=5):
        super(AntonyCnn, self).__init__()

        self.features = nn.Sequential(
            BasicConv2d(3, 32, kernel_size=11, padding=5, stride=2), # 112x112
            nn.MaxPool2d(3, padding=1, stride=2), # 56x56

            BasicConv2d(32, 64, kernel_size=5, padding=2, stride=1), # 56x56
            nn.MaxPool2d(3, padding=1, stride=2), # 28x28

            BasicConv2d(64, 96, kernel_size=3, padding=1, stride=1), # -> 28x28
            nn.MaxPool2d(3, padding=1, stride=2), # -> 14x14
    
            BasicConv2d(96, 128, kernel_size=3, padding=1, stride=1), # -> 14x14
            nn.Dropout2d(p=0.2),

            nn.MaxPool2d(3, padding=1, stride=2), # -> 7x7
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(
            nn.Linear(7*7*128, 1024),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        # x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def densenet121_model(num_class, use_pretrained = False):
    model = models.densenet121(pretrained = use_pretrained)

    in_features = model.classifier.in_features
    model.classifier = torch.nn.Linear(in_features, num_class)

    return model 

def densenet161_model(num_class, use_pretrained = False):
    model = models.densenet161(pretrained = use_pretrained)

    in_features = model.classifier.in_features
    model.classifier = torch.nn.Linear(in_features, num_class)

    return model
