import torch
import torch.nn as nn
import torch.nn.functional as F

#https://learn.microsoft.com/en-us/windows/ai/windows-ml/tutorials/pytorch-train-model

# Define a convolution neural network
class Network(nn.Module):

    x1 = None
    x2 = None
    x3 = None
    x4 = None

    def __init__(self,x1,x2,x3,x4):
        super(Network, self).__init__()

        self.x1 = x1
        self.x2 = x2
        self.x3 = x3
        self.x4 = x4
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=x1, kernel_size=5, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(x1)
        self.conv2 = nn.Conv2d(in_channels=x1, out_channels=x2, kernel_size=5, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(x2)
        self.pool = nn.MaxPool2d(2,2)
        self.conv4 = nn.Conv2d(in_channels=x2, out_channels=x3, kernel_size=5, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(x3)
        self.conv5 = nn.Conv2d(in_channels=x3, out_channels=x4, kernel_size=5, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(x4)
        self.fc1 = nn.Linear(x4 * 10 * 10, 10) # Sistemare padding, e uscita di linear e kernel_size

    def forward(self, input):
        output = F.relu(self.bn1(self.conv1(input)))      
        output = F.relu(self.bn2(self.conv2(output)))     
        output = self.pool(output)                        
        output = F.relu(self.bn4(self.conv4(output)))     
        output = F.relu(self.bn5(self.conv5(output)))     
        output = output.view(-1, self.x4 * 10 * 10)
        output = self.fc1(output)

        return output

def saveModel(model, accuracy):
    path = "model/cifar_net" + str(accuracy) + ".pth"
    torch.save(model.state_dict(), path)

# Function to test the model with the test dataset and print the accuracy for the test images
