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

torch.manual_seed(config.seed)

def train():
    #################
    # Load Data
    #################
    train_loader = get_loader(config.train_dir, config.label_dir, config.batchSize, "train", config.img_size, config.classList, config.data_aug_list)
    valid_loader = get_loader(config.valid_dir, config.valid_label_dir, config.batchSize, "valid", config.img_size, config.classList)

    #################
    # Create Model
    #################
    model = createDeepLabv3(config.nb_classes, config.img_size, config.architecture, config.pretrained, config.classifier)
    # Parallelization
    if config.USE_PARALLELIZATION == True:
        model = torch.nn.DataParallel(model)
    model.to(config.DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learningRate)

    # Scheduler
    if config.USE_SCHEDULER :
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=config.SCHEDULER_PATIENCE, factor=config.SCHEDULER_FACTOR)

    # Initialization
    best_iou = [0]
    loss_values_train = []
    loss_values_val = []
    mIoU_train = [[]for i in range(config.nb_classes)]
    mIoU_val = [[]for i in range(config.nb_classes)]

    print("Starting Training")
    for epoch in range(1, config.epochs + 1):
        elemCount_train = 0
        train_loss = 0
        visualize_train = False
        visualize_valid = False
        count_intersection_train = np.zeros(config.nb_classes)
        count_union_train = np.zeros(config.nb_classes)

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
            intersection_arr, union_arr = get_batch_intersection_union(output, target, config.nb_classes)
            count_intersection_train += intersection_arr
            count_union_train += union_arr

            # Get Batch Loss
            loss = calculate_loss(config.lossF, output, target, config.class_weight)
            train_loss += loss.sum().item()
            elemCount_train += len(data)

            # Update parameters
            loss.backward()
            optimizer.step()

            # Visualization
            if visualize_train == False:
                if epoch % config.freq_save == 0:
                    visualization_step(output, target, os.path.join(config.OUTPUT_TRAINING_ROOT, "visualization"), epoch,
                                       config.nb_train_show, config.class_colors, "train", name, None)
                visualize_train = True

        #################
        # Validation
        #################
        model.eval()
        valid_loss = 0
        elemCount_valid = 0
        count_intersection_valid = np.zeros(config.nb_classes)
        count_union_valid = np.zeros(config.nb_classes)
        with torch.no_grad():
            for data, target, name, not_norm_img in valid_loader:
                # Get Data
                data, target = data.to(config.DEVICE), target.to(config.DEVICE)

                # Get prediction
                output = model(data)["out"]
                pred = torch.argmax(output, dim=1)

                # Get Intersection & Union values (for computing IoU)
                intersection_arr, union_arr = get_batch_intersection_union(output, target, config.nb_classes)
                count_intersection_valid += intersection_arr
                count_union_valid += union_arr

                # Get Batch Loss
                loss = calculate_loss(config.lossF, output, target, config.class_weight)
                valid_loss += loss.sum().item()
                elemCount_valid += len(data)

                # Visualization
                if visualize_valid == False:
                    if epoch % config.freq_save == 0:
                        visualization_step(output, target, os.path.join(config.OUTPUT_TRAINING_ROOT, "visualization"), epoch,
                                           config.nb_valid_show, config.class_colors, "val", name, None)
                    visualize_valid = True

        # Compute and save epoch loss
        train_loss /= elemCount_train
        valid_loss /= elemCount_valid
        loss_values_train.append(train_loss)
        loss_values_val.append(valid_loss)

        # Compute and save epoch IoU
        train_miou, valid_miou = compute_epoch_IoU(count_intersection_train, count_union_train, count_intersection_valid, count_union_valid, config.nb_classes)
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
        #if epoch % config.freq_save == 0:
        #    torch.save(model.state_dict(), os.path.join(
        #        config.OUTPUT_TRAINING_ROOT, str(epoch) + '.pt'))

        # Save loss and mIoU plots evolution over epochs
        output_path_loss, output_path_train, output_path_val = visualize_epoch_curves(
            epoch, config.OUTPUT_TRAINING_ROOT, loss_values_train, loss_values_val, mIoU_train, mIoU_val, config.nb_classes, config.class_labels)

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
