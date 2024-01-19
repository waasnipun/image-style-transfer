## Visidon application test

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
    - [How to setup](#how-to-setup)
    - [Usage](#usage)
        - [Training](#training)
        - [Inference](#inference)
3. [Results](#results)
4. [Improvements](#improvements)

## Introduction

I implemented an image translation solution using a variant of U-Net architecture for a dataset of 1207 paired images. U-Net's encoder-decoder model, makes it powerful for tasks like image translation. U-Net is versatile, efficient in training, and can handle arbitrary image sizes, making it a suitable choice for this project. Here, the input images are resized to 256x256 making it to run any arbitary size images.

In my implementation, I utilized the PyTorch library for building and training the image translation model based on the U-Net architecture. 

**Rationale behind the Proposed Solution:**

The rationale behind the solution is centered on leveraging the power of CNN networks for encoding the image and decoding it to a styled image. 

The fully convolutional neural network (CNN) is a powerful architecture for image processing tasks due to its ability to automatically learn hierarchical representations from input data. CNNs consist of layers with learnable filters that automatically extract features at different levels of abstraction. 

The encoder portion of U-Net efficiently captures hierarchical features of the input image through a series of convolutional layers with downsampling. The encoded highly abstract and compressed version of the input image essentially consists information about the image's content and structure. This helps in extracting both high-level and low-level features.

The decoder utilizes transposed convolutional layers for upsampling, allowing the network to reconstruct the image at a higher resolution while maintaining the learned features.

![Image](/images/architecture.png)
**Refrenece - [pyimagesearch](https://pyimagesearch.com/2021/11/08/u-net-training-image-segmentation-models-in-pytorch/)**

 The main packages and components used in the project include:

PyTorch:

I chose PyTorch due to its popularity in the deep learning community, my previous experience with it, and user-friendly design. Its dynamic computational graph allows for more flexibility during model development and debugging.

NumPy:

NumPy is used for handling numerical operations and transformations on the image data. Its efficient array operations contribute to faster processing.

Matplotlib:

I utilized Matplotlib to visualize the input images, target images, and the generated output. This aids in qualitative assessment and debugging during model development.

PIL (Python Imaging Library):

I used PIL for loading and saving images in the pre-processing and post-processing stages of the image translation task.

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

![Image](/images/sample.png)

### Improvements

Further improvements
- Finetuning the hyper parameters
- Use Data augmentation - helps the model t generalize
- Can be employed by using transfer learning based approach - specifically for the Encoder
- Use a deeper network