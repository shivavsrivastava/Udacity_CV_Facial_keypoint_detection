## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ############## Following Naimish's paper #########
        # General Note: Convolution Formula = (W-F+2P)/S + 1 
        #               W- Input, F- Filter, S- Stride, P- Padding
        
        # Requirements==========
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoint. Last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # Assuming input of 225x225 
        # Layer1 (input 1x225x225, output 32x110x110) ===================
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 5)
        # maxpool 
        self.pool = nn.MaxPool2d(2,2)
        self.drop1 = nn.Dropout(0.1)
        
        # Layer2 (input 32x110x110, output 64x54x54) ===================
        # 32 input, 64 output channels/feature maps, 3x3 square convolution kernel
        self.conv2 = nn.Conv2d(32, 64, 3)
        
        # Layer3 (input 64x54x54, output 128x26x26) ===================
        # 64 input, 128 output channels/feature maps, 3x3 square convolution kernel
        self.conv3 = nn.Conv2d(64, 128, 3)
        
        # Layer4 (input 128x26x26, output 256x12x12) ===================
        # 128 input, 256 output channels/feature maps, 3x3 square convolution kernel
        self.conv4 = nn.Conv2d(128, 256, 3)
        
        # Layer5 (input 256x12x12, output 512x6x6) ===================
        # 256 input, 512 output channels/feature maps, 1x1 square convolution kernel
        self.conv5 = nn.Conv2d(256, 512, 1)
        
        # Layer6 ====================
        self.fc1 = nn.Linear(512*6*6, 1360)
        self.drop2 = nn.Dropout(0.5)
        
        # Layer7 ====================
        self.fc2 = nn.Linear(1360, 680)
        
        # Layer8 ====================
        self.fc3 = nn.Linear(680, 136)
      
        
        

        
    def forward(self, x):
        ## x is the input image
        x = self.pool(F.relu(self.conv1(x)))
        x = self.drop1(x)
        
        x = self.pool(F.relu(self.conv2(x)))
        x = self.drop1(x)
        
        x = self.pool(F.relu(self.conv3(x)))
        x = self.drop1(x)
        
        x = self.pool(F.relu(self.conv4(x)))
        x = self.drop1(x)
        
        x = self.pool(F.relu(self.conv5(x)))
        x = self.drop1(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Dense Layers
        x = self.drop2(F.relu(self.fc1(x)))
        x = self.drop2(F.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x

    
class SmallNet(nn.Module):

    def __init__(self):
        super(SmallNet, self).__init__()
        
        ############## Following Naimish's paper #########
        # General Note: Convolution Formula = (W-F+2P)/S + 1 
        #               W- Input, F- Filter, S- Stride, P- Padding
        
        # Requirements==========
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoint. Last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # Assuming input of 225x225 
        # Layer1 (input 1x225x225, output 32x110x110) ===================
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 5)
        # maxpool 
        self.pool = nn.MaxPool2d(2,2)
        self.drop1 = nn.Dropout(0.1)
        
        # Layer2 (input 32x110x110, output 64x54x54) ===================
        # 32 input, 64 output channels/feature maps, 3x3 square convolution kernel
        self.conv2 = nn.Conv2d(32, 64, 3)
        
        # Layer3 (input 64x54x54, output 128x26x26) ===================
        # 64 input, 128 output channels/feature maps, 3x3 square convolution kernel
        self.conv3 = nn.Conv2d(64, 128, 3)
        
        # Layer4 (input 128x26x26, output 256x12x12) ===================
        # 128 input, 256 output channels/feature maps, 3x3 square convolution kernel
        self.conv4 = nn.Conv2d(128, 256, 3)
        
        # Layer5 ====================
        self.fc1 = nn.Linear(256*12*12, 1000)
        self.drop2 = nn.Dropout(0.5)
        
        # Layer6 ====================
        self.fc2 = nn.Linear(1000, 136)
        
       
        
        

        
    def forward(self, x):
        ## x is the input image
        x = self.pool(F.relu(self.conv1(x)))
        x = self.drop1(x)
        
        x = self.pool(F.relu(self.conv2(x)))
        x = self.drop1(x)
        
        x = self.pool(F.relu(self.conv3(x)))
        x = self.drop1(x)
        
        x = self.pool(F.relu(self.conv4(x)))
        x = self.drop1(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Dense Layers
        x = self.drop2(F.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x
