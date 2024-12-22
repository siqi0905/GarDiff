

<h1>Improving Virtual Try-On with Garment-focused Diffusion Models</h1>
<div>
    <a>Siqi Wan</a><sup>1</sup>,
    <a>Yehao Li</a><sup>2</sup>,
    <a>Jingwen Chen</a><sup>2</sup>,
    <a>Yingwei Pan</a><sup>2</sup>,
    <a>Ting Yao</a><sup>2</sup>,
    <a>Yang Cao</a><sup>1</sup>
    <a>Tao Mei</a><sup>2</sup>
</div>
<div>
    <sup>1</sup>University of Science and Technology of China; <sup>2</sup>HiDream.ai Inc
</div>
</br>
This is the official repository for the [Paper](https://arxiv.org/abs/2409.08258) "Improving Virtual Try-On with Garment-focused Diffusion Models"

## Overview


## Installation
Create a conda environment and install dependencies:
```
pip install -r requirements.txt
```
## Dataset
You can download the VITON-HD dataset from [here](https://github.com/xiezhy6/GP-VTON) <br>
For inference, the following dataset structure is required: <br>
```
test
|-- image
|-- masked_vton_img 
|-- warp-cloth
|-- cloth
|-- cloth_mask
```
## Inference
Please download the pre-trained model from [Google Link](https://drive.google.com/drive/folders/1rXnxHwG-OrDtm-c58OuhYj0m4dwGaYYE?usp=drive_link)
```
sh inf_gar.sh
```
## Acknowledgement
Thanks the contribution of [LaDI-VTON](https://github.com/miccunifi/ladi-vton) and [GP-VTON](https://github.com/xiezhy6/GP-VTON).
