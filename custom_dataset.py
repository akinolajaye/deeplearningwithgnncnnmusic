
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import cv2
import numpy as np
import torch.nn as nn
from torch_geometric.nn import MLP,global_max_pool
import torchvision.models as models

PATCH_SIZE=1024
class CustomDataset(Dataset):
    def __init__(self, music_directory):
        self.file_paths = []
        # self.transform = transform
        self.processed_files = []

        for directory, _, files in os.walk(music_directory):
            png_files = [f for f in files if f.endswith('.png') and f not in self.processed_files]
            if png_files:
                self.file_paths += [os.path.join(directory, f) for f in png_files]
                self.processed_files += png_files

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        file_path = self.file_paths[index]
        file_name = self.processed_files[index]
        # Load image in grayscale
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        
        height, width = image.shape  # Corrected shape unpacking for a grayscale image
        random_width = np.random.choice(width - PATCH_SIZE, size=2)
        
        patch1 = image[:, random_width[0]:random_width[0] + PATCH_SIZE]
        patch2 = image[:, random_width[1]:random_width[1] + PATCH_SIZE]
        
        # Normalize the pixel values of each patch
        patch1 = torch.tensor(patch1).unsqueeze(0).float() / 127.5 - 1  # Add channel dimension
        patch2 = torch.tensor(patch2).unsqueeze(0).float() / 127.5 - 1  # Add channel dimension
        file_index = torch.tensor(index)
        
        # Adding a dummy dimension for compatibility
        file_index = file_index.unsqueeze(-1)

        
        # No need to permute since we're not converting to (C, H, W) format, as it's already a single-channel image
        return patch1, patch2, file_index, file_name.replace(".png", "")



class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
    

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride=2)
        self.pool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Linear(512, 128)
        self.mlp = MLP([128, 256, 32], norm=None)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward_once(self, x):
        # Forward pass for one input
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        x_pooled=x
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.mlp(x)
        return x ,x_pooled

    def forward(self, x1, x2):
        # Forward pass for both inputs
        output1 , output1_cnn = self.forward_once(x1)
        output2, output2_cnn = self.forward_once(x2)
        return output1, output2 ,output1_cnn, output2_cnn

    

        
resnet50 = models.resnet50(pretrained=True)        
class CustomResNet50(nn.Module):
    def __init__(self):
        super(CustomResNet50, self).__init__()

        
        self.resnet50 = nn.Sequential(*(list(resnet50.children())[:-1]))
        self.pool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(1024, 512)
        self.mlp = MLP([512, 256, 32], norm=None)

        # # Example additional layers
        # self.additional_layers = nn.Sequential(
        #     nn.AdaptiveMaxPool2d ((1, 1)),  # Add adaptive pooling
        #     nn.Linear(2048, 1024),  # First additional FC layer
        #     nn.Linear(1024, 512),
        #     MLP([512, 256, 32], norm=None)
        #     # nn.ReLU(),
        #     # nn.Dropout(0.5),
        #     # nn.Linear(512, output_features)  # Final output layer
        # )
        
    def forward(self, x1,x2):
        output1 , output1_cnn = self.forward_once(x1)
        output2, output2_cnn = self.forward_once(x2)
        return output1, output2 ,output1_cnn, output2_cnn
        
    def forward_once(self, x):
        # Forward pass for one input
        x = self.resnet50(x)
        x_pooled =x
        
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        # x = self.fc2(x)
        x= self.mlp(x)

        return x ,x_pooled
        
        
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.out = GCNConv(hidden_channels, num_classes)  # num_classes is the number of classes for classification

    def forward(self, x, edge_index):
        # First layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        # Second layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        # Third layer
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        # Output layer
        x = self.out(x, edge_index)
        return F.log_softmax(x, dim=1)