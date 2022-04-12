import os
import argparse
import torch
import numpy as np
import hiyapyco
from shutil import copyfile

from loader import get_loader
from models.create_model import create_model
from utils.compute_IoU import get_batch_intersection_union, compute_epoch_IoU, save_IoU_epoch, print_IoU
from utils.compute_loss import get_loss_weights, get_loss, calculate_loss, print_losses
from utils.visualizer import (visualization_step, visualize_epoch_curves)


def train(args):
    # Read config file
    config_path = args.config
    config = hiyapyco.load(config_path, method=hiyapyco.METHOD_MERGE)
    
    # Set seed & device
    torch.manual_seed(config["seed"])
    device = config["logic"]["device"]
    
    # Create output folder & copy config file
    training_output_root = config["paths"]["output"]["training"]
    os.makedirs(training_output_root, exist_ok=True)
    os.makedirs(os.path.join(training_output_root, "visualization"), exist_ok=True)
    copyfile(config_path, training_output_root+"/config.yml")
    
    #################
    # Load Data
    #################
    train_loader = get_loader(
        config = config,
        state = "train"
    )
    valid_loader = get_loader(
        config = config,
        state = "validation"
    )

    #################
    # Create Model
    #################
    model = create_model(config)
    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config["logic"]["training"]["lr"]
    )

    # Scheduler
    if config["logic"]["training"]["scheduler"]["use_schedule"] :
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            verbose=True, 
            patience=config["logic"]["training"]["scheduler"]["patience"], 
            factor=config["logic"]["training"]["scheduler"]["factor"]
        )

    #################
    # Training
    #################
    nb_classes = len(config["logic"]["classes"])
    nb_epochs = config["logic"]["training"]["epochs"]

    # Initialization
    best_iou = [0]
    loss_values_train = []
    loss_values_val = []
    mIoU_train = [[]for i in range(nb_classes)]
    mIoU_val = [[]for i in range(nb_classes)]

    # Parameters
    loss_weights = get_loss_weights(config).to(device)
    criterion = get_loss(
        function = config["logic"]["training"]["loss"]["function"],
        weights = loss_weights
    )

    print("Starting Training")
    for epoch in range(1, nb_epochs + 1):
        elemCount_train = 0
        train_loss = 0
        count_intersection_train = np.zeros(nb_classes)
        count_union_train = np.zeros(nb_classes)

        model.train()
        for data, target, name in train_loader:
            # Get data
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            output = model(data)["out"]
            pred = torch.argmax(output, dim=1)

            # Get Intersection & Union values (for computing IoU)
            intersection_arr, union_arr = get_batch_intersection_union(output, target, nb_classes)
            count_intersection_train += intersection_arr
            count_union_train += union_arr

            # Get Batch Loss
            loss = calculate_loss(
                criterion, 
                output, 
                target
            )
            train_loss += loss.sum().item()
            elemCount_train += len(data)

            # Update parameters
            loss.backward()
            optimizer.step()

            # Visualization
            if epoch % config["logic"]["visualization"]["show"]["frequency"] == 0:
                visualization_step(
                    output = output, 
                    target = target, 
                    output_dir = os.path.join(training_output_root, "visualization"), 
                    epoch = epoch,
                    nb_show = config["logic"]["visualization"]["show"]["train"], 
                    class_colors_dict = config["logic"]["classes"], 
                    state = "train", 
                    name = name
                )

        #################
        # Validation
        #################
        model.eval()
        valid_loss = 0
        elemCount_valid = 0
        count_intersection_valid = np.zeros(nb_classes)
        count_union_valid = np.zeros(nb_classes)
        with torch.no_grad():
            for data, target, name in valid_loader:
                # Get Data
                data, target = data.to(device), target.to(device)

                # Get prediction
                output = model(data)["out"]
                pred = torch.argmax(output, dim=1)

                # Get Intersection & Union values (for computing IoU)
                intersection_arr, union_arr = get_batch_intersection_union(output, target, nb_classes)
                count_intersection_valid += intersection_arr
                count_union_valid += union_arr

                # Get Batch Loss
                loss = calculate_loss(
                    criterion, 
                    output, 
                    target
                )
                valid_loss += loss.sum().item()
                elemCount_valid += len(data)

                # Visualization
                if epoch % config["logic"]["visualization"]["show"]["frequency"] == 0:
                    visualization_step(
                        output = output, 
                        target = target, 
                        output_dir = os.path.join(training_output_root, "visualization"), 
                        epoch = epoch,
                        nb_show = config["logic"]["visualization"]["show"]["validation"], 
                        class_colors_dict = config["logic"]["classes"], 
                        state = "val", 
                        name = name
                    )

        # Compute and save epoch loss
        train_loss /= elemCount_train
        valid_loss /= elemCount_valid
        loss_values_train.append(train_loss)
        loss_values_val.append(valid_loss)

        # Compute and save epoch IoU
        train_miou, valid_miou = compute_epoch_IoU(count_intersection_train, count_union_train, count_intersection_valid, count_union_valid, nb_classes)
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
            torch.save(model.state_dict(), os.path.join(training_output_root, 'best.pt'))

        # Save model by epoch
        #if epoch % config.SAVE_FREQUENCY == 0:
        #    torch.save(model.state_dict(), os.path.join(
        #        config.OUTPUT_TRAINING_ROOT, str(epoch) + '.pt'))

        # Save loss and mIoU plots evolution over epochs
        visualize_epoch_curves(
            epoch = epoch, 
            output_dir = training_output_root, 
            loss_values_train = loss_values_train, 
            loss_values_val = loss_values_val, 
            mIoU_train = mIoU_train, 
            mIoU_val = mIoU_val, 
            considered_classes = nb_classes, 
            class_labels = list(config["logic"]["classes"])
        )

        # Print metrics for this epoch
        print_losses(epoch, train_loss, valid_loss)
        print_IoU(train_miou, valid_miou)

        # Update learning rate
        if config["logic"]["training"]["scheduler"]["use_schedule"]:
            scheduler.step(valid_loss)

    print("Best IoU is :", best_iou, best_valid_iou)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Segmentation model !')
    parser.add_argument('config', default="config_files/multiclass/config.py")
    args = parser.parse_args()
    train(args)
