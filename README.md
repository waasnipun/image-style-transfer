## Visidon application test

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
    - [How to setup](#how-to-setup)
    - [Usage](#usage)
        - [Training](#training)
        - [Inference](#inference)
3. [Results](#results)

## Introduction

I implemented an image translation solution using PyTorch and U-Net architecture for a dataset of 1207 paired images. U-Net's encoder-decoder model, with skip connections, preserves fine details, making it powerful for tasks like image translation. U-Net is versatile, efficient in training, and can handle arbitrary image sizes, making it a suitable choice for this project.

## Getting Started

Follow these steps to get started with the project:

### How to setup

1. Install the required dependencies:

    ```bash
    python
    torch
    torchvision
    numpy
    matplotlib
    scikit-learn
    Pillow
   ```

   if you have pip run - `pip install -r requirements.txt`

2. Copy the VD_dataset2 dataset folder to the root directory. Directory tree should look like below.
```
├── VD_dataset2 
│   ├── 0000_input.png
│   ├── 0000_target.png
│   │   .
│   │   .
│   │   .
│   │   
│   └── 4824_target.png
├── output
├── model.py
├── notebooks 
│   ├── inference.ipynb
│   └── train.ipynb
├── requirement.txt
├── model.py
├── tools.py
├── train.py
├── inference.py
├── trained _model
│   ├── efficientnet_trained.pth
│   ├── resnet18.pth
│   └── vit_trained.pth
├── test_image.png
└── README.md
```


### Usage

1. **Training:**
    - Run the `python3 train.py` to train the U-Net model.

2. **Inferencing:**
    - Run the `python3 inference.py --image_path test_image.png` command replacing the the image for translation


### Results

Result evaluating in the test set

![Image](/output/sample.png)