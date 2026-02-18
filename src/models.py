import torch.nn as nn
from torchvision.models import resnet18

#def get_resnet18(num_classes=10, pretrained=False):
    #"""
    #Returns a ResNet-18 model.
    #Even though Task A only uses 5 classes, the output head is initialized for 10 classes
    #to accommodate the full Continual Learning scenario across both tasks.
    #"""
    ## Instantiates a standard ResNet-18 backbone
   # model = resnet18(weights='IMAGENET1K_V1' if pretrained else None)
    #
    ## Replaces the final fully connected layer to match the target number of classes
   # num_ftrs = model.fc.in_features
   # model.fc = nn.Linear(num_ftrs, num_classes)
    #
  #  return model


def get_resnet18(num_classes=10, pretrained=False):  # Defaults to 10 output classes to cover both continual learning tasks
    """
    Returns a ResNet-18 model adapted for CIFAR-10 (32x32 images).
    """
    # Instantiates the base ResNet-18 backbone, optionally loading ImageNet-pretrained weights
    model = resnet18(weights='IMAGENET1K_V1' if pretrained else None)

    # Critical adaptation for CIFAR-10: replaces the standard 7x7 kernel (stride 2),
    # designed for large ImageNet images, with a 3x3 kernel (stride 1) that preserves
    # the original 32x32 spatial resolution and prevents excessive downsampling.
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

    # Removes the initial max-pooling layer, which would reduce a 32x32 feature map
    # to 16x16 prematurely; replaced with an identity mapping to retain spatial detail.
    model.maxpool = nn.Identity()

    # Replaces the final fully connected layer to produce logits for the target number of classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model
