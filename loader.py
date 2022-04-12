import os
import torch
import torchvision
from PIL import Image

def get_loader(config, state, drop_last=False):
    """Returns data loader
    """
    img_dir = config["paths"][state]["img"]
    lab_dir = config["paths"][state]["label"]
    batch_size = config["logic"]["training"]["batch_size"]
    img_size = config["logic"]["training"]["img_size"]
    classList = config["logic"]["classes"]
    data_aug_list = config["logic"]["training"]["data_aug"]
    
    if lab_dir != None :
        dataset = DataLoaderLabel(img_dir, lab_dir, state, img_size, classList, data_aug_list)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=drop_last)
    else :
        # Use for unannotated data (test)
        dataset = DatasetTest(img_dir)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=2)

    print("Number of images for", state, ":", len(dataset))

    return loader


class DataLoaderLabel(torch.utils.data.Dataset):
    def __init__(self, img_dir, lab_dir, state, img_size, classList, data_aug_list=[]):
        self.img_dir = img_dir
        self.lab_dir = lab_dir
        self.state = state
        self.resize = img_size
        self.classList = classList
        self.data_aug_list = data_aug_list

        self.fileList = [
            f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f)) and ('.png' in f or ".jpg" in f or ".JPG" in f)]
        self.indexes = range(len(self.fileList))
        self.dic = {k: self.fileList[k] for k in self.indexes}
        self.get_class_color_Tensor()

    def get_class_color_Tensor(self):
        classes_color = []
        for seg_class in self.classList:
            r,g,b = seg_class["r"], seg_class["g"], seg_class["b"]
            color_normalized = [int(r)/255, int(g)/255, int(b)/255]
            classes_color.append(color_normalized)
        self.classesTensor = torch.Tensor(classes_color)

    def __len__(self):
        return len(self.fileList)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        idx = self.indexes[idx]

        # Read image
        img_name = os.path.join(self.img_dir, self.dic[idx])
        img_extension = img_name.split(".")[-1]
        image = Image.open(img_name).convert('RGB')
        image.verify()

        # Read label
        ref_name = os.path.join(self.lab_dir, self.dic[idx].replace(img_extension, "png"))
        imageRef = Image.open(ref_name).convert('RGB')
        imageRef.verify()

        # Data augmentation
        if self.data_aug_list != [] and self.state == "train":
            mirror = torch.rand((2,))
            if "hflip" in self.data_aug_list:
                if mirror[0] >= 0.5:
                    image = torchvision.transforms.functional.hflip(image)
                    imageRef = torchvision.transforms.functional.hflip(imageRef)
            if "vflip" in self.data_aug_list:
                if mirror[1] >= 0.5:
                    image = torchvision.transforms.functional.vflip(image)
                    imageRef = torchvision.transforms.functional.vflip(imageRef)

        # Resize
        image = torchvision.transforms.functional.resize(
            image, size=[self.resize, self.resize])
        imageRef = torchvision.transforms.functional.resize(
            imageRef, size=[self.resize, self.resize], interpolation=0)

        # To Tensor
        image = (torchvision.transforms.functional.to_tensor(image))
        labelImage = torchvision.transforms.functional.to_tensor(imageRef)

        # Normalize image
        image_normalized = torchvision.transforms.functional.normalize(
            image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Prepare label
        labelFinal = torch.zeros(labelImage.size()[1:], dtype=torch.long)
        torch.set_printoptions(profile="full")
        labelImage = labelImage.permute(1, 2, 0) # Axis reordered

        for i in range(self.classesTensor.size(0)):
            labelFinal += (labelImage == self.classesTensor[i]).all(dim=-1) * i

        return image_normalized, labelFinal, self.dic[idx].split(".")[0]


class DatasetTest(torch.utils.data.Dataset):
    # DataLoader for testing when there are no labels
    def __init__(self, train_dir, img_size, classList):
        self.train_dir = train_dir
        self.fileList = [
            f for f in os.listdir(train_dir) if os.path.isfile(os.path.join(train_dir, f))  and ('.png' in f or ".jpg" in f or ".JPG" in f)]
        self.indexes = range(len(self.fileList))
        self.dic = {k: self.fileList[k] for k in self.indexes}
        self.resize = img_size
        self.classList = classList

    def __len__(self):
        return len(self.fileList)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        idx = self.indexes[idx]
        img_name = os.path.join(self.train_dir, self.dic[idx])
        image = Image.open(img_name).convert('RGB')
        image.verify()

        # Resize
        image = torchvision.transforms.functional.resize(
            image, size=[self.resize, self.resize])

        # To Tensor
        image = (torchvision.transforms.functional.to_tensor(image))

        # Normalize
        image_normalized = torchvision.transforms.functional.normalize(
            image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        return image_normalized, self.dic[idx].split(".")[0] , image