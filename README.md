# GarDiff
This repository is the official implementation of GarDiff
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
