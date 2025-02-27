import torch
import torch.nn as nn
from torchvision import models


class_names = ['F_Breakage', 'F_Crushed', 'F_Normal', 'R_Breakage', 'R_Crushed', 'R_Normal']
num_classes = len(class_names)

class car_classifier_resnet(nn.Module): 
    def __init__(self, num_classes):  
        super().__init__()
        self.model_res = models.resnet50(weights='DEFAULT')

        # Freeze all layers except the final fully connected layer
        for param in self.model_res.parameters():
            param.requires_grad = False
            
        # Unfreeze layer4 and fc layers
        for param in self.model_res.layer4.parameters():
            param.requires_grad = True  

        in_features = self.model_res.fc.in_features

        self.model_res.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes)

        )
    
    def forward(self, x):
        return self.model_res(x)