import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN3D(nn.Module):
    def __init__(self, input_shape, num_classes):
        """
        Initialize a 3D CNN model.

        Args:
            input_shape (tuple): Shape of the input tensor (C, D, H, W).
            num_classes (int): Number of output classes.
        """
        super(CNN3D, self).__init__()
        
        self.conv1  = nn.Conv3d(in_channels=input_shape[0], out_channels=32, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.bn1    = nn.BatchNorm3d(32)
        self.pool1  = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)

        self.conv2  = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.bn2    = nn.BatchNorm3d(64)
        self.pool2  = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)

        self.conv3  = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.bn3    = nn.BatchNorm3d(128)
        self.pool3  = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)

        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc          = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.global_pool(x)
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        x = self.fc(x)
        return x