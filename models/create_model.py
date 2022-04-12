import torchvision
import torchvision.models
import torch.nn as nn

def create_model(config):
    out_channels = len(config["logic"]["classes"])
    img_size = config["logic"]["training"]["img_size"]
    architecture = config["logic"]["training"]["architecture"]
    pretrained = config["logic"]["training"]["pretrained"]
    freeze_weight = config["logic"]["training"]["freeze_weight"]

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
        raise

    # Freeze weights if base architecture is pretrained
    if freeze_weight :
        for param in model.parameters():
            param.requires_grad = False

    # Replace end of base architecture with classifier
    print("Using simple convolution for classifier")
    model.classifier[4] = nn.Conv2d(in_channels=in_chan,
                                    out_channels=out_channels,
                                    kernel_size=1,
                                    stride=1)
    for param in model.classifier[4].parameters():
        param.requires_grad = True
    return model