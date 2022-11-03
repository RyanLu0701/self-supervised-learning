from torchvision import datasets, models
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class restnet50(nn.Module):
    def __init__(self,):
        super(restnet50, self).__init__()

        self.model = models.resnet50(pretrained = False)

        self.fc1 = nn.Linear(1000, 512)
        self.fc2 = nn.Linear(512, 4)

    def forward(self,x,label = False):

        x = self.model(x)
        x = self.fc1(x)
        if label == True:


            return self.fc2(x)

        return x
