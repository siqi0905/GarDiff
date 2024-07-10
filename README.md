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
```
cd src
sh inf_gar.sh
```
## Acknowledgement
We borrow some codes from [LaDI-VTON](https://github.com/miccunifi/ladi-vton). We also thank [GP-VTON](https://github.com/xiezhy6/GP-VTON), our warping module depends on it.
