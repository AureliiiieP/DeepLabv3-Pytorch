import os, sys
from shutil import copyfile

import cv2
import numpy as np
import torch
import CONFIG
sys.path.append(CONFIG.config_file)
import config

from loader import get_loader
from models.create_model import createDeepLabv3
from utils.compute_IoU import get_batch_intersection_union, compute_epoch_IoU, save_IoU_epoch, print_IoU
from utils.compute_loss import calculate_loss, print_losses
from utils.visualizer import (visualization_step,
                                          visualize_epoch_curves)

torch.manual_seed(config.SEED)

def train():
    #################
    # Load Data
    #################
    train_loader = get_loader(config.TRAIN_IMG_DIR, config.TRAIN_LABEL_DIR, config.BATCH_SIZE, "train", config.IMG_SIZE, config.CLASS_LIST, config.DATA_AUG_LIST)
    valid_loader = get_loader(config.VALID_IMG_DIR, config.VALID_LABEL_DIR, config.BATCH_SIZE, "valid", config.IMG_SIZE, config.CLASS_LIST)

    #################
    # Create Model
    #################
    model = createDeepLabv3(config.NB_CLASSES, config.IMG_SIZE, config.ARCHITECTURE, config.PRETRAINED, config.CLASSIFIER_HEAD)
    # Parallelization
    if config.USE_PARALLELIZATION == True:
        model = torch.nn.DataParallel(model)
    model.to(config.DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)

    # Scheduler
    if config.USE_SCHEDULER :
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=config.SCHEDULER_PATIENCE, factor=config.SCHEDULER_FACTOR)

    # Initialization
    best_iou = [0]
    loss_values_train = []
    loss_values_val = []
    mIoU_train = [[]for i in range(config.NB_CLASSES)]
    mIoU_val = [[]for i in range(config.NB_CLASSES)]

    print("Starting Training")
    for epoch in range(1, config.EPOCHS + 1):
        elemCount_train = 0
        train_loss = 0
        count_intersection_train = np.zeros(config.NB_CLASSES)
        count_union_train = np.zeros(config.NB_CLASSES)

        #################
        # Training
        #################
        model.train()
        for data, target, name, _ in train_loader:
            # Get data
            data, target = data.to(config.DEVICE), target.to(config.DEVICE)
            optimizer.zero_grad()

            output = model(data)["out"]
            pred = torch.argmax(output, dim=1)

            # Get Intersection & Union values (for computing IoU)
            intersection_arr, union_arr = get_batch_intersection_union(output, target, config.NB_CLASSES)
            count_intersection_train += intersection_arr
            count_union_train += union_arr

            # Get Batch Loss
            loss = calculate_loss(config.LOSS_FUNCTION, output, target, config.CLASS_WEIGHT)
            train_loss += loss.sum().item()
            elemCount_train += len(data)

            # Update parameters
            loss.backward()
            optimizer.step()

            # Visualization
            if epoch % config.SAVE_FREQUENCY == 0:
                visualization_step(output, target, os.path.join(config.OUTPUT_TRAINING_ROOT, "visualization"), epoch,
                                    config.TRAIN_VISUALIZATION_NB_IMG_SHOW, config.CLASS_COLORS, "train", name, None)

        #################
        # Validation
        #################
        model.eval()
        valid_loss = 0
        elemCount_valid = 0
        count_intersection_valid = np.zeros(config.NB_CLASSES)
        count_union_valid = np.zeros(config.NB_CLASSES)
        with torch.no_grad():
            for data, target, name, not_norm_img in valid_loader:
                # Get Data
                data, target = data.to(config.DEVICE), target.to(config.DEVICE)

                # Get prediction
                output = model(data)["out"]
                pred = torch.argmax(output, dim=1)

                # Get Intersection & Union values (for computing IoU)
                intersection_arr, union_arr = get_batch_intersection_union(output, target, config.NB_CLASSES)
                count_intersection_valid += intersection_arr
                count_union_valid += union_arr

                # Get Batch Loss
                loss = calculate_loss(config.LOSS_FUNCTION, output, target, config.CLASS_WEIGHT)
                valid_loss += loss.sum().item()
                elemCount_valid += len(data)

                # Visualization
                if epoch % config.SAVE_FREQUENCY == 0:
                    visualization_step(output, target, os.path.join(config.OUTPUT_TRAINING_ROOT, "visualization"), epoch,
                                        config.VALID_VISUALIZATION_NB_IMG_SHOW, config.CLASS_COLORS, "val", name, None)


        # Compute and save epoch loss
        train_loss /= elemCount_train
        valid_loss /= elemCount_valid
        loss_values_train.append(train_loss)
        loss_values_val.append(valid_loss)

        # Compute and save epoch IoU
        train_miou, valid_miou = compute_epoch_IoU(count_intersection_train, count_union_train, count_intersection_valid, count_union_valid, config.NB_CLASSES)
        mIoU_train = save_IoU_epoch(train_miou, mIoU_train)
        mIoU_val = save_IoU_epoch(valid_miou, mIoU_val)

        #################
        # Print & Save
        #################

        # Save model by best
        if best_iou[0] < np.mean(valid_miou):
            print("Model saved")
            best_valid_iou = valid_miou
            best_iou = [np.mean(valid_miou)]            
            torch.save(model.state_dict(), os.path.join(config.OUTPUT_TRAINING_ROOT, 'best.pt'))

        # Save model by epoch
        #if epoch % config.SAVE_FREQUENCY == 0:
        #    torch.save(model.state_dict(), os.path.join(
        #        config.OUTPUT_TRAINING_ROOT, str(epoch) + '.pt'))

        # Save loss and mIoU plots evolution over epochs
        output_path_loss, output_path_train, output_path_val = visualize_epoch_curves(
            epoch, config.OUTPUT_TRAINING_ROOT, loss_values_train, loss_values_val, mIoU_train, mIoU_val, config.NB_CLASSES, config.CLASS_LABELS)

        # Print metrics for this epoch
        print_losses(epoch, train_loss, valid_loss)
        print_IoU(train_miou, valid_miou)

        # Update learning rate
        if config.USE_SCHEDULER:
            scheduler.step(valid_loss)

    print("Best IoU is :", best_iou, best_valid_iou)

if __name__ == "__main__":
    os.makedirs(config.OUTPUT_TRAINING_ROOT, exist_ok=True)
    os.makedirs(os.path.join(config.OUTPUT_TRAINING_ROOT, "visualization"), exist_ok=True)
    copyfile(os.path.join(CONFIG.config_file, "config.py"), config.OUTPUT_TRAINING_ROOT+"/config.py")
    train()
