"""Functions related to loss calculation
"""
import torch

def get_loss_weights(config):
    weight_list = list(config["logic"]["training"]["loss"]["weights"])
    weights = []
    for seg_class in weight_list :
        weights.append(config["logic"]["training"]["loss"]["weights"][seg_class])
    weight_tensor = torch.Tensor(weights)
    return weight_tensor

def get_loss(function, weights):
    if function == "CE":
        criterion = torch.nn.CrossEntropyLoss(weight=weights, reduction='mean')
    else :
        print("Not implemented loss.")
        raise
    return criterion

def calculate_loss(criterion, output, target):
    loss = criterion(output, target)
    return loss

def print_losses(epoch, train_loss, valid_loss):
    print("Epoch " + str(epoch) + "  :  train_Loss=" + str(train_loss) + "    val_Loss=" + str(valid_loss))