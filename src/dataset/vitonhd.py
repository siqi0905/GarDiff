# File heavily based on https://github.com/aimagelab/dress-code/blob/main/data/dataset.py

import json
import os
import pickle
import random
import sys
from pathlib import Path
from typing import Tuple, Literal
import torchvision.transforms.functional as TF

PROJECT_ROOT = Path(__file__).absolute().parents[2].absolute()
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageDraw

class VitonHDDataset(data.Dataset):
    def __init__(
        self,
        dataroot_path: str,
        phase: Literal["train", "test"],
        order: Literal["paired", "unpaired"] = "paired",
        size: Tuple[int, int] = (512, 384),
    ):
        super(VitonHDDataset, self).__init__()
        self.dataroot = dataroot_path
        self.phase = phase
        self.height = size[0]
        self.width = size[1]
        self.size = size


        self.norm = transforms.Normalize([0.5], [0.5])
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.transform2D = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )
        self.toTensor = transforms.ToTensor()

        self.order = order

        self.toTensor = transforms.ToTensor()

        im_names = []
        c_names = []
        dataroot_names = []


        if phase == "train":
            filename = os.path.join(dataroot_path, f"{phase}_pairs.txt")
        else:
            filename = os.path.join(dataroot_path, f"{phase}_pairs.txt")

        with open(filename, "r") as f:
            for line in f.readlines():
                if phase == "train":
                    im_name, _,_ = line.strip().split()
                    c_name = im_name
                else:
                    if order == "paired":
                        im_name, _ ,_= line.strip().split()
                        c_name = im_name
                    else:
                        im_name, c_name,_ = line.strip().split()

                im_names.append(im_name)
                c_names.append(c_name)
                dataroot_names.append(dataroot_path)

        self.im_names = im_names
        self.c_names = c_names
        self.dataroot_names = dataroot_names
        self.flip_transform = transforms.RandomHorizontalFlip(p=1)
    def __getitem__(self, index):
        c_name = self.c_names[index]
        im_name = self.im_names[index]
        
        
        cloth = Image.open(os.path.join(self.dataroot, self.phase, "cloth", c_name.replace('.png','.jpg')))

        im_pil_big = Image.open(
            os.path.join(self.dataroot, self.phase, "image", im_name.replace('.png','.jpg'))
        ).resize((self.width,self.height))

        image = self.transform(im_pil_big)

        if self.phase == 'test':
            warped_cloth = Image.open(
                os.path.join(self.dataroot, self.phase, "warp_cloth_u", c_name.replace('.jpg','.png'))
            ).resize((self.width,self.height))

        else:    
            warped_cloth = Image.open(
                os.path.join(self.dataroot, self.phase, "warp_cloth", c_name.replace('.jpg','.png'))
            ).resize((self.width,self.height))

        warped_cloth = self.transform(warped_cloth)
       
        # load parsing image


        mask = Image.open(os.path.join(self.dataroot, self.phase, "agnostic-densepose-v3", im_name.replace('.png','_mask.jpg'))).resize((self.width,self.height))
        mask = self.toTensor(mask)
        mask = mask[:1]
      


        mask = 1-mask

       
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1

        im_mask = image * mask

       
        result = {}
        result["c_name"] = c_name
        result["image"] = image
        # result["cloth"] = cloth_trim
        result["cloth"] = self.transform(cloth)
        result['warped_cloth'] = warped_cloth
        result["inpaint_mask"] = 1-mask
        result["im_mask"] = im_mask
        result["captions"] = "model is wearing an upper body garment" 
        result["im_name"] = im_name
        result["category"] = self.order

        return result

    def __len__(self):
        return len(self.im_names)
