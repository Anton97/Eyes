
import torch
import torch.nn as nn
from torchvision import models

def load_model(weights_path: str, num_classes: int = 2):
    model = models.resnet34(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    model.eval()
    return model
