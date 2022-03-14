# Semantic segmentation

Semantic segmentation


## Prepare your data

Note. Images should be jpg or png. Labels should be png

### (Optional) Make patches ###
Depending on your data, you may want to create patches out of your images.
e.g. If you have huge images (satellite images etc)

## Training the model

### Step 1. Fill out the config file ###

Create your config file set the path in the CONFIG.py file.
An example config.file is provided.

### Step 2. Train model ###

```
python train.py
```

### Step 3. Test model ###
```
python test.py
```
Note. fp16 available for small gain of inference speed

## Folder organization example ##

Example of how to organize the files for the scripts to work

```
.
├── config_files
│   ├── binary_segmentation
│   │    └── config.py
│   └── multiclass_segmentation
│        └── config.py
├── data 
│   ├── train 
│   │       ├── img
│   │       │     ├── IMG_5686.JPG
│   │       │     └──...
│   │       └── label
│   │             ├── IMG_5686.png
│   │             └──...
│   ├── valid
│   └── test
├── utils
├── CONFIG.py
├── loader.py 
├── test.py 
├── train.py 
└── README.md (this file)
```

## Example on toy dataset ##
### Toy dataset - Pizza topping ###
Imagine you are a pizza restaurant. You need to make sure the toppings match the customer order and that they cover a decent surface on the dough.

image
contact if want to download dataset & annotation

### Training ###
Can monitor IoU, loss over epochs
Also visualize regularly prediciton on val to get a sense of model training.

### Example of results ###
image of inference result on toy dataset

