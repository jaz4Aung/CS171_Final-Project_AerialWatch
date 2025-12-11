
# AerialWatch: Human Presence Detection from Drone Images
CS 171 – Fall 2025 Final Project

Authors:

- Aung Aung – Custom CNN model, data preparation, analysis & visualization

- Margarita Rincon – Transfer learning model (ResNet/MobileNet), data processing, evaluation
## Project Overview

The goal of this project is to determine whether deep learning models can reliably detect the presence or absence of humans in aerial drone images.
Unlike traditional image classification tasks, humans in drone imagery often appear extremely small, making the problem challenging for lightweight neural networks.

We explore this question by training and analyzing:

- A custom CNN built from scratch
- A ResNet18 model using transfer learning

The analysis notebook focuses on model behavior, including success cases, failure cases, and generalization to external drone images.

## Installation Instructions
This project uses PyTorch, Torchvision, and common scientific Python packages.

Install dependencies:

```bash
pip install torch torchvision
pip install numpy pandas matplotlib pillow


(Optional) For Mac M1/M2 GPU acceleration with Metal:

pip install torch==<mps-compatible-version>
```

Clone the repository:
```bash
git clone <my-repo-url>
cd AerialWatch
```

Launch Jupyter Notebook:
```bash
jupyter notebook
```
## Data Access

Our dataset contains aerial RGB images from SARD Search & Rescue datasets and extra images from internet. These images were collected from real drone flights and include:
- Natural outdoor scenes
- High variation in camera altitude and angle
- Small human figures that occupy only a few pixels
- Many cluttered or low-visibility environments

Images are organized into the following directory structure:
```code
data/
 ├── train/
 │     └── images/
 ├── valid/
 │     └── images/
 └── test/
       └── images/
```
Each split has a corresponding CSV file:
train_data.csv, 
valid_data.csv, 
test_data.csv.

| Column         | Meaning                               |
|----------------|----------------------------------------|
| image_filename | filename inside the `/images/` directory |
| has_person     | 1 = human present, 0 = no humans        |

This format ensures reproducibility and allows the dataset to be loaded on any operating system.

Download Link for main dataset: https://drive.google.com/file/d/1DqctU2uuiGeLeU-YNASPM8zLB2nIVOq9/view?usp=drive_link

## Notebook Component
Model Construction 

- Builds and trains the custom CNN

- Includes architecture explanation

- Shows training/validation loss curves

- Examines class imbalance

- Loads and preprocesses dataset

Analysis & Visualization Notebook

- Shows the model in action

- Visualizes predictions from the CNN

- Displays misclassified images

- Evaluates generalization on external drone images

- Answers the core research question


## Future Work

If we were to continue this project, we would pursue the following improvements:

- Train on higher resolution images (e.g., 512×512)
Small humans become nearly invisible when downsampled to 224×224.

- Balance the dataset
The current dataset has far more human images, creating a prediction bias.

- Expand external evaluation
Test on diverse drone footage from real-world missions.

- Implement confidence calibration
Reduce false positives in empty terrain.

- Use object detection models (YOLO, Faster R-CNN)
Whole-image classification is limited for tiny humans; detection models can localize people more reliably.

- Deploy a lightweight model on a drone platform
(e.g., TensorRT, MobileNet, model quantization).