import os
import argparse
import torch
import numpy as np
import hiyapyco
from shutil import copyfile

from loader import get_loader
from models.create_model import create_model
from utils.compute_IoU import get_batch_intersection_union, compute_epoch_IoU, compute_test_IoU, save_IoU_epoch, print_IoU
from utils.compute_loss import get_loss_weights, get_loss, calculate_loss, print_losses
from utils.visualizer import (visualization_step, visualize_epoch_curves)


def test(args):
    # Read config file
    config_path = args.config
    config = hiyapyco.load(config_path, method=hiyapyco.METHOD_MERGE)
    
    # Set seed & device
    torch.manual_seed(config["seed"])
    device = config["logic"]["device"]

    # Create output folder & copy config file
    output_test_dir = config["paths"]["output"]["inference"]
    os.makedirs(output_test_dir, exist_ok=True)
    copyfile(config_path, output_test_dir+"/config.yml")
    
    #################
    # Load Data
    #################
    test_loader = get_loader(
        config = config,
        state = "test",
        drop_last = False
    )

    #################
    # Create Model
    #################
    model = create_model(config)
    saved_weights = config["paths"]["test"]["pretrained_weights"]
    model.load_state_dict(torch.load(saved_weights))
    model.eval()
    model.to(device)

    elem_count = 0
    if config["paths"]["test"]["use_label"] == True:
        nb_classes = len(config["logic"]["classes"])
        count_intersection_test = np.zeros(nb_classes)
        count_union_test = np.zeros(nb_classes)
        with torch.no_grad():
            for data, target, name in test_loader:
                data, target = data.to(device), target.to(device)
                
                output = model(data)["out"]
                pred = torch.argmax(output, dim=1)
                
                # Get Intersection & Union values (for computing IoU)
                intersection_arr, union_arr = get_batch_intersection_union(output, target, nb_classes)
                count_intersection_test += intersection_arr
                count_union_test += union_arr
                
                elem_count += len(data)

                # Save inference
                visualization_step(
                    output = output, 
                    target = target, 
                    output_dir = output_test_dir, 
                    epoch = None,
                    nb_show = 1, 
                    class_colors_dict = config["logic"]["classes"], 
                    state = "test", 
                    name = name
                )
            test_miou = compute_test_IoU(count_intersection_test, count_union_test, nb_classes)

        # Print mIoU for training and validation for the current epoch
        print("Test IoU = ", test_miou)
        np.savetxt(os.path.join(output_test_dir, "IoU.txt"), test_miou, fmt="%s")
    else:
        with torch.no_grad():
            for data, name, unnorm_data in test_loader:
                data = data.to(device)

                output = model(data)["out"]
                elem_count += len(data)
                pred = torch.argmax(output, dim=1)
                #torch.cuda.current_stream().synchronize() #Use if want to get time spend doing inference

                visualization_step(
                    output = output, 
                    target = None, 
                    output_dir = output_test_dir, 
                    epoch = None,
                    nb_show = 1, 
                    class_colors_dict = config["logic"]["classes"], 
                    state = "test", 
                    name = name
                )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Segmentation model !')
    parser.add_argument('config', default="config_files/multiclass/config.py")
    args = parser.parse_args()
    test(args)
