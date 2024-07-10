from pathlib import Path
import sys
from PIL import Image
import os
import torch
import cv2

PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference_hd import GardiffHD
from src.inference_dc import GardiffDC


import argparse
parser = argparse.ArgumentParser(description='run ootd')
parser.add_argument('--gpu_id', '-g', type=int, default=0, required=False)
parser.add_argument('--dataroot', type=str, default="", required=True)
parser.add_argument('--model_type', type=str, default="hd", required=False)
parser.add_argument('--category', '-c', type=int, default=0, required=False)
parser.add_argument('--scale', type=float, default=2.0, required=False)
parser.add_argument('--step', type=int, default=20, required=False)
parser.add_argument('--sample', type=int, default=4, required=False)
parser.add_argument('--seed', type=int, default=-1, required=False)
args = parser.parse_args()


category_dict = ['upperbody', 'lowerbody', 'dress']
category_dict_utils = ['upper_body', 'lower_body', 'dresses']

if args.model_type == "hd":
    model = GardiffHD(args.gpu_id)
elif args.model_type == "dc":
    model = GardiffDC(args.gpu_id)
else:
    raise ValueError("model_type must be \'hd\' or \'dc\'!")


if __name__ == '__main__':


    filename = os.path.join(args.dataroot, f"test_pairs.txt")


    im_names = []
    c_names = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            im_name, c_name,_= line.strip().split()
            im_names.append(im_name)
            c_names.append(c_name)
    for i in range(len(im_names)):

        mask = Image.open(os.path.join(args.dataroot, 'mask',c_names[i])).convert('RGB')
        cloth_img = Image.open(os.path.join(args.dataroot, 'cloth',c_names[i]).replace('.png','.jpg')).resize((384,512))        
        model_img = Image.open(os.path.join(args.dataroot, 'image',im_names[i].replace('.png','.jpg'))).resize((384,512))
        warp_cloth = Image.open(os.path.join(args.dataroot, 'warp_cloth',c_names[i])).resize((384,512))
        masked_vton_img = Image.open(os.path.join(args.dataroot, 'masked_vton_img',im_names[i].replace('.png','.jpg'))).resize((384,512))
        warp_image = Image.open(os.path.join(args.dataroot, 'mask',c_names[i]))

        images = model(
            model_type=args.model_type,
            category=category_dict[args.category],
            image_garm=cloth_img,
            image_vton=masked_vton_img,
            warp_cloth=warp_cloth,
            mask=mask,
            image_ori=model_img,
            num_samples=args.n_samples,
            num_steps=args.n_steps,
            image_scale=args.image_scale,
            seed=args.seed,
        )

        image_idx = 0
        for image in images:
            
            image.save(os.path.join('result', im_names[i]))
            image_idx += 1