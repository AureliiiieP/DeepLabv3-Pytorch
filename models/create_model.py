import torchvision
import torchvision.models
import torch.nn as nn
from .deeplabv3 import DeepLabHead

def createDeepLabv3(outputchannels, img_size, architecture, pretrained=False, classifier="simple", freeze_weight = False):
    # https://discuss.pytorch.org/t/removing-classification-layer-for-resnet101-deeplabv3/51004/2

    # Create model with specified base architecture
    if architecture == "deeplabv3_resnet101":
        model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=pretrained)
        in_chan = 256
    elif architecture == "deeplabv3_resnet50":
        model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=pretrained)
        in_chan = 256
    elif architecture == "fcn_resnet101":
        model = torchvision.models.segmentation.fcn_resnet101(pretrained=pretrained)
        in_chan = 512
    elif architecture == "fcn_resnet50":
        model = torchvision.models.segmentation.fcn_resnet50(pretrained=pretrained)
        in_chan = 512
    else :
        print("Not a valid architecture")

    # Freeze weights if base architecture is pretrained
    if freeze_weight :
        for param in model.parameters():
            param.requires_grad = False

    # Replace end of base architecture with classifier
    if classifier == "simple":
        print("Using simple convolution for classifier")
        model.classifier[4] = nn.Conv2d(in_channels=in_chan,
                                        out_channels=outputchannels,
                                        kernel_size=1,
                                        stride=1)
        for param in model.classifier[4].parameters():
            param.requires_grad = True
    elif classifier == "deeplabhead":
        print("Using DeepLabHead classifier")
        model.classifier = DeepLabHead(2048, outputchannels, img_size)
        for param in model.classifier.parameters():
            param.requires_grad = True
    else:
        print("Not a valid classifier head.")

    return model