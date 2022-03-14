"""Functions related to loss calculation
"""
import torch

def print_losses(epoch, train_loss, valid_loss):
    print("Epoch " + str(epoch) + "  :  train_Loss=" + str(train_loss) + "    val_Loss=" + str(valid_loss))

def calculate_loss(lossF, output, target, class_weight):
    if lossF == "CE":
        criterion = torch.nn.CrossEntropyLoss(weight=class_weight, reduction='mean')
    else :
        print("Not implemented loss.")
    loss = criterion(output, target)
    return loss