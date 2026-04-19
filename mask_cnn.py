import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 32 * 32, 128)  # 128x128 image -> after pooling twice = 32x32
        self.fc2 = nn.Linear(128, 2)  # 2 classes: mask, no mask

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # conv1 + relu + pool
        x = self.pool(F.relu(self.conv2(x)))  # conv2 + relu + pool
        x = x.view(-1, 64 * 32 * 32)          # flatten
        x = F.relu(self.fc1(x))               # fc1 + relu
        x = self.fc2(x)                       # fc2
        return x