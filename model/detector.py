import torch
from torch import nn


class circle_detector(nn.Module):
    def __init__(self):
        super(circle_detector, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, 3), nn.BatchNorm2d(64), nn.ReLU(), #64*198*198
            nn.MaxPool2d(2), # 64*99*99
            nn.Conv2d(64, 128, 3), nn.BatchNorm2d(128), nn.ReLU(), #128*97*97
            nn.MaxPool2d(2), # 128*48*48 
            nn.Conv2d(128, 128, 3), nn.BatchNorm2d(128), nn.ReLU(), #128*46*46
            nn.MaxPool2d(2), #128*23*23
            nn.Conv2d(128, 128, 3), nn.BatchNorm2d(128), nn.ReLU(), #128*21*21
            nn.MaxPool2d(2), #128*10*10
            nn.Conv2d(128, 32, 1), nn.BatchNorm2d(32), nn.ReLU(), #32*10*10
        )
        self.fc_layer = nn.Sequential(nn.Linear(3200, 100), nn.ReLU(), nn.Linear(100, 20), nn.ReLU(), nn.Linear(20,3))
    
    def forward(self, x):
        x = self.conv_layers(x)
        B, _, _, _ = x.shape
        x = self.fc_layer(x.view(B, -1))
        return x
