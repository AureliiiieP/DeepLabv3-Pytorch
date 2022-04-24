"""Functions related to IoU computation
"""
import torch
import numpy as np

def print_IoU(train_miou, valid_miou):
    print("Training IoU = ", train_miou, " / Validation IoU = ", valid_miou)

def save_IoU_epoch(epoch_miou, mIoU_epoch):
    """Saves mIoU of current epoch (epoch_miou) to later plot evolution of mIoU over epochs
    """
    for i in range(len(epoch_miou)):
        mIoU_epoch[i].append(epoch_miou[i])
    return mIoU_epoch

def get_inter_union_pixels_single_img(pred, label, C, EMPTY=1., ignore=None):
    """Array of IoU for each (non ignored) class for a single image (not batch)
    """
    union_array = np.zeros(C)   
    intersection_array = np.zeros(C)
    for i in range(C):
        if i != ignore:
            intersection = ((label == i) & (pred == i)).sum()
            union = ((label == i) | ((pred == i) )).sum()
            count_label = (label == i).sum()
            intersection_array[i]=intersection
            union_array[i]=union
    return intersection_array, union_array

def get_batch_intersection_union(output, target, nb_classes):
    """Get Intersection & Union values (for computing IoU)
    """
    intersection_arr = np.zeros(nb_classes)
    union_arr = np.zeros(nb_classes)
    for i in range(0, len(output)):
        pred = torch.argmax(output.cpu(), dim=1)[i].detach().numpy()
        _intersection_arr, _union_arr = get_inter_union_pixels_single_img(
            pred, target[i].cpu().numpy(), nb_classes)
        intersection_arr += _intersection_arr
        union_arr += _union_arr
    return intersection_arr, union_arr

def compute_epoch_IoU(count_intersection_train, count_union_train, count_intersection_valid, count_union_valid, nb_classes):
    """Compute train and validation IoU for one epoch given union and intersection values
    """
    # Train IoU
    train_miou = np.zeros(nb_classes)
    for i in range(nb_classes):
        if count_union_train[i] == 0 :
            train_miou[i] = 0
        else:
            train_miou[i] = count_intersection_train[i]/count_union_train[i]

    # Valid IoU
    valid_miou = np.zeros(nb_classes)
    for i in range(nb_classes):
        if count_union_valid[i] == 0 :
            valid_miou[i] = 0
        else:
            valid_miou[i] = count_intersection_valid[i]/count_union_valid[i]

    return train_miou, valid_miou

def compute_test_IoU(count_intersection_test, count_union_test, nb_classes):
    """Compute train and validation IoU for one epoch given union and intersection values
    """
    # Test IoU
    test_miou = np.zeros(nb_classes)
    for i in range(nb_classes):
        if count_union_test[i] == 0 :
            test_miou[i] = 0
        else:
            test_miou[i] = count_intersection_test[i]/count_union_test[i]

    return test_miou