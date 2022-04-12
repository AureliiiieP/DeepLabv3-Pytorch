import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_colors(class_colors_dict):
    colors = []
    for color in class_colors_dict :
        class_color = (color["r"], color["g"], color["b"])
        colors.append(class_color)
    return colors

def visualize_epoch_curves(epoch, output_dir, loss_values_train, loss_values_val, mIoU_train, mIoU_val, considered_classes, class_labels):
    epoch_plot = np.arange(epoch)

    output_path_loss = os.path.join(output_dir,"loss_curve.jpg")
    output_path_train = os.path.join(output_dir,"mIoU_curve_train.jpg")
    output_path_val = os.path.join(output_dir,"mIoU_curve_val.jpg")

    visualize_loss(epoch_plot, loss_values_train, loss_values_val, output_path_loss)
    visualize_mIoU(epoch_plot, mIoU_train, output_path_train, considered_classes, class_labels)
    visualize_mIoU(epoch_plot, mIoU_val, output_path_val, considered_classes, class_labels)

def visualize_mIoU(epoch_plot, data, output_path, considered_classes, class_labels):
    """
    Function that plots the mIoU per class
    """
    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(considered_classes):
        mIou_plot = np.arange(len(data[i]))
        ax.plot(epoch_plot, data[i], label=list(class_labels[i])[0])
    ax.legend()
    plt.show()
    plt.savefig(output_path)
    plt.close()

def visualize_loss(epoch_plot, loss_values_train, loss_values_val, output_path):
    """
    Function that plots the train and validation loss
    """
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(epoch_plot, np.array(loss_values_train), label='Train Loss')
    ax.plot(epoch_plot, np.array(loss_values_val), label='Validation Loss')
    ax.legend()
    plt.show()
    plt.savefig(output_path)
    plt.close()

def visualize_output(output, i, class_colors):
    """
    Given the batch output of the model and the index of the image to save inside the batch,
    returns pred - Matrix containing the predicted class for each pixel of the image
    Returns :
            visualization_image - Predicted cv2 image
    """
    pred = torch.argmax(output.cpu(), dim=1)[i].detach().numpy()
    visualization_image = np.zeros((pred.shape[0], pred.shape[1],3), np.uint8)
    for x in range(pred.shape[0]):
        for y in range(pred.shape[1]):
            visualization_image[x, y]=class_colors[pred[x, y]]
    visualization_image = cv2.cvtColor(visualization_image, cv2.COLOR_BGR2RGB)
    return pred, visualization_image

def save_prediction(blank_image, target, output_dir, state, name):
    """
    Given a predicted image, reference image, saves them to the output_dir
    """
    fname_pred_val = os.path.join(output_dir, name + "_pred_" + state +".jpg")
    fname_ref_val = os.path.join(output_dir, name + "_ref_" + state + ".jpg")
    cv2.imwrite(fname_pred_val, blank_image)
    
    plt.imshow(target.cpu())
    plt.savefig(fname_ref_val)
    return fname_pred_val, fname_ref_val

def visualization_step(output, target, output_dir, epoch, nb_show, class_colors_dict, state, name):
    """
    This function go through the predictions of a model and saves them as images in output_dir :

    Parameters - output : Output from the model of a batch of image
               - output_dir : Path to the directory in which the images will be saved
               - target : The Batch of reference images (annotated img)
               - nb_show : The number of images to save
               - class_colors_dict : Dict of the colors (RGB) to use when displaying prediction.
                                Index of the color in the array should correspond to the label of the associated class.
    """
    class_colors = get_colors(class_colors_dict)
    for i in range(0, nb_show):
        pred, blank_image = visualize_output(output, i, class_colors)
        # Save images
        name_img = str(epoch) + "_" + name[i]
        fname_pred, fname_ref = save_prediction(blank_image, target[i], output_dir, state, name_img)