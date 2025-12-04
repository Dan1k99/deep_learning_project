import torch.nn as nn
from torchvision.models import resnet18

def get_resnet18(num_classes=10, pretrained=False):
    """
    Returns a ResNet-18 model.
    Even though Task A only uses 5 classes, we initialize output for 10 
    to handle the Continual Learning scenario[cite: 22].
    """
    # Instantiate standard ResNet-18 [cite: 21]
    model = resnet18(weights='IMAGENET1K_V1' if pretrained else None)
    
    # Modify the final Fully Connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model