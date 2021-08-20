# Hybrid Routing Transformer for Zero-Shot Learning

## Overview
This repository is the official pytorch implementation of Hybrid Routing Transformer for Zero-Shot Learning.  

## Requirements


Experiments were done with the following package versions for Python 3.6:

- PyTorch 1.4.0 with CUDA 10.0
- Numpy 1.16.4
- Tensorboard 1.10.0
- Tensorflow 1.2.0
- H5py 2.9.0
- Pandas 0.24.2
- Torchvision 0.5.0



## Data Preparation

1. Please download and data into the `./data folder.` We show the details about download links.  

2. Please run the feature extraction scripts in the `./extract folder.` to get the feature maps of Resnet101.

## Train and Test

For different datasets (AWA2/CUB/SUN), you can run the code:

```
python HRT_AWA2.py 
python HRT_CUB.py 
python HRT_SUN.py

``` 

