import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ResNet18Classifier(nn.Module):
    def __init__(self, nr_classes: int) -> None:
        super(ResNet18Classifier, self).__init__()

        # Load pretrained ResNet50
        self.base_model = models.resnet18(pretrained=True)

        # Freeze the feature extractor 
        for param in self.base_model.parameters():
             param.requires_grad = True

       
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(in_features, nr_classes)

    def forward(self, x):
        return self.base_model(x)  
    
    