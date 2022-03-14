import os
import torch

SEED = 2019
DEVICE = torch.device("cuda")
USE_PARALLELIZATION = False

#############################
# Training
#############################

OUTPUT_TRAINING_ROOT = "output/training/20220314_Pizza_topping" #For Train

# Training and Validation data folders
TRAIN_IMG_DIR = "data/train/img"
TRAIN_LABEL_DIR = "data/train/label"
VALID_IMG_DIR = "data/val/img"
VALID_LABEL_DIR = "data/val/label"

#############################
# INFERENCE
#############################

OUTPUT_TEST_ROOT = 'output/test/20220314_Pizza_topping' #For Test
saved_weights = "" #For Test
# Testing data folders
test_dir = 'data/test'
test_lab_dir = 'data/test' # If no label are available, put at None
quantization = "fp16" #Using quantization may improve speed at a small cost of accuracy

#############################
# Model parameters
#############################

# Labels & class weights
CLASS_COLORS = [(0, 0, 0),(255, 128, 128),(255, 255, 0),(255, 255, 255),(128, 64, 64),(255, 0, 0)]
CLASS_LIST = torch.Tensor([[0, 0, 0],[1.0, 128/255, 128/255],[1.0, 1.0, 0],[1.0, 1.0, 1.0],[128/255, 64/255, 64/255],[1.0, 0, 0]])
CLASS_WEIGHT = torch.Tensor([1, 1, 1, 1, 1, 1]).to(DEVICE)
CLASS_LABELS = ["Background","Dough","Cheese","Mozzarella","Tomato Sauce","Prosciutto",]

# Architecture
ARCHITECTURE = "deeplabv3_resnet50" # Choose from deeplabv3_resnet50, deeplabv3_resnet101, fcn_resnet50 and fcn_resnet101
CLASSIFIER_HEAD = "simple"  # Choose from deeplabhead or simple
PRETRAINED = False
DATA_AUG_LIST = ["hflip"] #Possibilities : "hflip", "vflip". Put at [] if you don't want any

# Training parameters
BATCH_SIZE = 4
LR = 0.001
IMG_SIZE = 512
EPOCHS = 300
NB_CLASSES = 6

# Loss
LOSS_FUNCTION = "CE" 

# Scheduler parameters
USE_SCHEDULER = False
SCHEDULER_PATIENCE = 80
SCHEDULER_FACTOR = 0.8

# Display images and frequency during training
TRAIN_VISUALIZATION_NB_IMG_SHOW = 2 # Must be smaller than batch size
VALID_VISUALIZATION_NB_IMG_SHOW = 2 # Must be smaller than batch size
SAVE_FREQUENCY = 10