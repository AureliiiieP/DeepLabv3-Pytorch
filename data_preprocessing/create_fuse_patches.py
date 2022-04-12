import os
import glob
import hiyapyco
import argparse
import cv2
import numpy as np
from PIL import Image

def create_patches(img_path, output_path, patch_size):
    # Load image
    img_name = os.path.basename(img_path).split(".")[0]
    img = cv2.imread(img_path)

    # Calculate maximum number of patches we can get without resizing
    nb_patches_x = img.shape[0]//patch_size
    nb_patches_y = img.shape[1]//patch_size
    print(f"Creating {nb_patches_x} x {nb_patches_y} patches")
    #img = cv2.resize(img, (nb_patches_x*patch_size, nb_patches_y*patch_size), interpolation = cv2.INTER_AREA)
    #cv2.imwrite(os.path.join(output_path,"resize.png"), img)

    # Create patches
    idx = 0
    for j in range(nb_patches_x):
        for i in range(nb_patches_y):
            patch = img[j*patch_size:(j+1)*patch_size, i*patch_size:(i+1)*patch_size]
            cv2.imwrite(os.path.join(output_path,img_name + "_patch_" + str(idx)+".png"), patch)
            idx +=1
    # Add last row (might have overlap)
    for i in range(nb_patches_y):
        patch = img[img.shape[0]-patch_size:img.shape[0], i*patch_size:(i+1)*patch_size]
        cv2.imwrite(os.path.join(output_path,img_name + "_patch_" + str(idx)+".png"), patch)
        idx +=1
    # Add last column(might have overlap)
    for i in range(nb_patches_x):
        patch = img[i*patch_size:(i+1)*patch_size,img.shape[1]-patch_size:img.shape[1]]
        cv2.imwrite(os.path.join(output_path,img_name + "_patch_" + str(idx)+".png"), patch)
        idx +=1
    # Add last patch (bottom left)
    patch = img[img.shape[0]-patch_size:img.shape[0],img.shape[1]-patch_size:img.shape[1]]
    cv2.imwrite(os.path.join(output_path,img_name + "_patch_" + str(idx)+".png"), patch)

def fuse_patches(folder_path, img_name, output_path, original_size, patch_size):
    image = Image.new('RGB', original_size)
    nb_patches_x = original_size[0]//patch_size
    nb_patches_y = original_size[1]//patch_size
    for file_path in glob.glob(os.path.join(folder_path,"*.png")): 
        if img_name == os.path.basename(file_path).split("__")[0].split("_patch_")[0] :
            im = Image.open(file_path)
            img_id = int(file_path.split(img_name+"_patch_")[1].split("__")[0].split(".")[0])
            if img_id<nb_patches_x*nb_patches_y:
                column = img_id//nb_patches_x
                row = img_id%nb_patches_x
                image.paste(im, (row*patch_size, column*patch_size))
            elif  img_id>=nb_patches_x*nb_patches_y and img_id<nb_patches_x*nb_patches_y+nb_patches_x:
                position = img_id-(nb_patches_x*nb_patches_y)
                image.paste(im, (position*patch_size,original_size[1]-patch_size))
            elif  img_id>=nb_patches_x*nb_patches_y+nb_patches_x and img_id<nb_patches_x*nb_patches_y+nb_patches_x+nb_patches_y:
                position = img_id-(nb_patches_x*nb_patches_y+nb_patches_x)
                image.paste(im, (original_size[0]-patch_size,position*patch_size))
            else :
                image.paste(im, (original_size[0]-patch_size,original_size[1]-patch_size))
    image.save(os.path.join(output_path,img_name+'.jpg'), quality=95)

def create_patches_multiple_img(folder_path, output_path, crop_size):
    # Create output folder
    os.makedirs(output_path, exist_ok=True)
    # Loop over all images in the folder
    for f in os.listdir(folder_path):
        file_path = os.path.join(folder_path,f)
        create_patches(file_path, output_path, crop_size)

def fuse_multiple_img(folder_path, output_path, original_size, patch_size):
    # Create output folder
    os.makedirs(output_path, exist_ok=True)
    # Match all patch to their original image based on their name
    list_img_name = {}
    for f in os.listdir(folder_path):
        name = f.split("__")[0].split("_patch_")[0]
        list_img_name[name]=1
    for img_name in list_img_name:
        if "." not in img_name :
            print("Processing :", img_name)
            fuse_patches(folder_path, img_name, output_path, original_size, patch_size)

def main(args):
    config_path = args.config
    config = hiyapyco.load(config_path, method=hiyapyco.METHOD_MERGE)

    mode = config["logic"]["preprocessing"]["mode"]
    if mode == "crop":
        input_path = config["paths"]["preprocessing"]["make_patches"]["input"]
        output_path = config["paths"]["preprocessing"]["make_patches"]["output"]
        crop_size = config["logic"]["preprocessing"]["patch_size"]
        create_patches_multiple_img(input_path, output_path, crop_size)
    elif mode == "fuse" :
        input_path = config["paths"]["preprocessing"]["fuse_patches"]["input"]
        output_path = config["paths"]["preprocessing"]["fuse_patches"]["output"]
        crop_size = config["logic"]["preprocessing"]["patch_size"]
        original_size_width = config["logic"]["preprocessing"]["original_size"]["width"]
        original_size_height = config["logic"]["preprocessing"]["original_size"]["height"]
        fuse_multiple_img(input_path, output_path, (original_size_width, original_size_height), crop_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Segmentation model !')
    parser.add_argument('config', default="config_files/multiclass/config.py")
    args = parser.parse_args()
    main(args)