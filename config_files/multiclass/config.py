import os
import torch

seed = 2019
DEVICE = torch.device("cuda")
USE_PARALLELIZATION = False

#############################
# Training
#############################

OUTPUT_TRAINING_ROOT = "output/training/20220314_Pizza_topping" #For Train

# Training and Validation data folders
train_dir = "data/train/img"
label_dir = "data/train/label"
valid_dir = "data/val/img"
valid_label_dir = "data/val/label"

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
class_colors = [(0, 0, 0),(255, 128, 128),(255, 255, 0),(255, 255, 255),(128, 64, 64),(255, 0, 0)]
classList = torch.Tensor([[0, 0, 0],[1.0, 128/255, 128/255],[1.0, 1.0, 0],[1.0, 1.0, 1.0],[128/255, 64/255, 64/255],[1.0, 0, 0]])
class_weight = torch.Tensor([1, 1, 1, 1, 1, 1]).to(DEVICE)
class_labels = ["Background","Dough","Cheese","Mozzarella","Tomato Sauce","Prosciutto",]

# Architecture
classifier = "simple"  # Choose from deeplabhead or simple
architecture = "deeplabv3_resnet50" # Choose from deeplabv3_resnet50, deeplabv3_resnet101, fcn_resnet50 and fcn_resnet101
pretrained = False
data_aug_list = ["hflip"] #Possibilities : "hflip", "vflip". Put at [] if you don't want any

# Training parameters
batchSize = 4
learningRate = 0.001
img_size = 512
epochs = 300
nb_classes = 6

# Loss
lossF = "CE" 

# Scheduler parameters
USE_SCHEDULER = False
SCHEDULER_PATIENCE = 80
SCHEDULER_FACTOR = 0.8

# Display images and frequency during training
nb_train_show = 2 # Must be smaller than batch size
nb_valid_show = 2 # Must be smaller than batch size
freq_save = 10
conf_matrix_freq = 10