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

I implemented an image translation solution using U-Net architecture for a dataset of 1207 paired images. U-Net's encoder-decoder model, makes it powerful for tasks like image translation. U-Net is versatile, efficient in training, and can handle arbitrary image sizes, making it a suitable choice for this project.

In my implementation, I utilized the PyTorch library for building and training the image translation model based on the U-Net architecture. PyTorch is a powerful deep learning framework known for its flexibility, dynamic computation graph, and ease of use. The main packages and components used in the project include:

1. **PyTorch:**
   - **Description:** PyTorch is an open-source machine learning library that provides dynamic computation graphs, making it well-suited for building and training neural networks.
   - **Rationale:** I chose PyTorch due to its popularity in the deep learning community and  my previous experience with pytorch.

2. **U-Net Architecture:**
   - **Description:** U-Net is a convolutional neural network architecture designed for tasks like image segmentation. It consists of an encoder-decoder structure with skip connections.
   - **Rationale:** The U-Net architecture was chosen for its ability to capture both global and local features, making it suitable for image translation tasks. 

**Rationale behind the Proposed Solution:**

The rationale behind the solution is centered on leveraging established and effective tools in the deep learning domain, emphasizing ease of development, flexibility, and the ability to handle arbitrary image sizes. The chosen libraries and packages collectively contribute to a comprehensive and efficient solution for image translation.

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

### Improvements

Further improvements
- Finetuning the hyper parameters
- Use Data augmentation - helps the model t generalize
- Can be employed by using transfer learning based approach - specifically for the Encoder
- Use a deeper network