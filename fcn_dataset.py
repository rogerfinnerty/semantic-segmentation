"""CS585 HW4"""

import os
import csv
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
rev_normalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])

def paired_crop_and_resize(image, label, size):
    """Random crop"""
    i, j, h, w = transforms.RandomResizedCrop.get_params(image, scale=(0.5, 1.0), ratio=(1, 1))
    image = transforms.functional.resized_crop(image, i, j, h, w, size, Image.BILINEAR)
    label = transforms.functional.resized_crop(label, i, j, h, w, size, Image.NEAREST)

    return image, label

def paired_resize(image, label, size):
    image = transforms.functional.resize(image, size, Image.BILINEAR)
    label = transforms.functional.resize(label, size, Image.NEAREST)

    return image, label

class CamVidDataset(Dataset):
    """Class for video dataset"""
    def __init__(self, root, images_dir, labels_dir, class_dict_path, resolution, crop=False):
        self.root = root
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.resolution = resolution
        self.crop = crop
        self.class_dict = self.parse_class_dict(os.path.join(root, class_dict_path))
        self.images = [os.path.join(root, images_dir, img) for img in sorted(os.listdir(os.path.join(root, images_dir)))]
        self.labels = [os.path.join(root, labels_dir, lbl) for lbl in sorted(os.listdir(os.path.join(root, labels_dir)))]

    def __len__(self):
        """ samples in dataset """
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        label_path = self.labels[idx]
        image = Image.open(image_path).convert("RGB")
        label = Image.open(label_path).convert("RGB")
        label = self.rgb_to_class_id(label)
        if self.crop:
            image, label = paired_crop_and_resize(image, label, self.resolution)
        else:
            image, label = paired_resize(image, label, self.resolution)

        # image to tensor and normalize
        image = transforms.ToTensor()(image)
        image = normalize(image)
        label = torch.tensor(np.array(label)).long()
        return image, label

    def parse_class_dict(self, class_dict_path):
        """return a dictionary that maps class id (0-31) to a tuple ((R,G,B), class_name)"""
        class_dict = {}
        with open(class_dict_path, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            for _ in range(1):
                next(csv_reader)

            for idx, row in enumerate(csv_reader):
                class_name = row[0]
                r, g, b = int(row[1]), int(row[2]), int(row[3])
                class_dict[idx] = ((r,g,b), class_name)

        # raise NotImplementedError("Implement the method")
        return class_dict

    def rgb_to_class_id(self, label_img):
        """Convert an RGB label image to a class ID image (H, W, 3) -> (H, W)"""
        label_img = np.array(label_img) # convert to numpy
        h, w = label_img.shape[0], label_img.shape[1]
        class_id_img = np.zeros((h,w)) # initialize class img
        for row in range(h):
            for col in range(w):
                (r,g,b) = label_img[row,col,:]  # extract RGB value of pixel
                for class_id, (color, _) in self.class_dict.items(): # match RGB values to class in dictionary
                    if color == (r,g,b):
                        class_id_img[row,col] = class_id

        # raise NotImplementedError("Implement the method")
        return Image.fromarray(class_id_img)

if __name__ == "__main__":
    images_dir = "train/"
    labels_dir = "train_labels/"
    class_dict_path = "class_dict.csv"
    resolution = (240, 240)
    camvid_dataset = CamVidDataset(root='CamVid/', images_dir=images_dir, labels_dir=labels_dir, class_dict_path=class_dict_path, resolution=resolution)

    # Example of loading a single sample
    image, label = camvid_dataset[0]

    # To visualize or further process, you might want to convert 'label' back to a color 
    # image or directly use it for training a segmentation model.
    label_vis = label.numpy().astype(np.float32)
    label_vis /= 31.
    label_vis *= 255.
    label_vis = label_vis.astype(np.uint8)
    label_vis = Image.fromarray(label_vis)
    label_vis.save("label_vis.png")
    image_vis = transforms.functional.to_pil_image(rev_normalize(image))
    image_vis.save("image_vis.png")
