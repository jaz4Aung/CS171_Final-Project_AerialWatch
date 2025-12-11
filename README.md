## Description
This is the Final Project for CS_171/SJSU/Fall_2025 class, which explores how convolutional neural networks (CNNs) can distinguish between aerial drone images that contain people and those that do not. Using RGB imagery from the VisDrone and SAR Search & Rescue datasets, we train and evaluate models that classify each image into “people” or “no people.” The goal is to develop a lightweight and efficient vision model suitable for real-time aerial analysis.

## Project Title
**AerialWatch: Detection of Human presence from Drone Images**

## Authors
**Aung Aung** , Github: jaz4Aung <br>
**Margarita Rincon**, Github: Mago-RM

## Question and Research Topic

We investigate whether deep learning models can reliably determine the **presence or absence of humans** in aerial RGB images captured by drones. Using drone imagery from datasets such as VisDrone, we frame the task as a **binary classification problem** with two classes: `people` and `no_people`.

A major challenge is that humans appear as **very small objects** in drone images due to altitude, camera angle, and resolution limits. These factors make the classification problem difficult because meaningful human features can be lost during image downsampling. To address this, we apply data augmentation and experiment with two different model architectures.

### Model Comparison
This project compares two approaches:

1. **MobileNetV2 (Transfer Learning)**  
   A pretrained backbone fine-tuned for binary classification, leveraging ImageNet features for improved robustness and generalization.

2. **Custom CNN (From Scratch)**  
   A lightweight convolutional neural network built and trained from scratch by our teammate, designed specifically for this task.

We evaluate which approach performs better on drone imagery and under what conditions pretrained models outperform smaller, task-specific networks.

### Evaluation
Performance is measured using:
- **Accuracy**
- **Precision & Recall**
- **Confusion Matrix**

We analyze common failure cases—such as occlusions, scale variation, and cluttered environments—and outline strategies to improve generalization, including higher-resolution inputs, stronger augmentation, and additional diverse training data.

## Project Outline
### Data Collection Plan

**Aung**: 
<ul>
<li>Dataset:Will be using the SARD search and rescue dataset for training, testing, and validations.</li>
<li>Source: https://universe.roboflow.com/datasets-pdabr/sard-8xjhy </li>
<li>Labelling: Relabel the images as the "people" and "no people categories" </li>
<li>Data Cleaning: Clean the data by removing duplicates and blurry photos, photo size adjustments and normalize the data.</li>
<li>Data splitting: Divide the dataset into training (70%), validation (15%), and testing (15%) subsets. </li>
<li>Additional Evaluation Data:<br>Collect offline drone images (not included in the SARD dataset) to serve as an external test set for assessing the model’s real-world performance after implementation. </li>
</ul>

**Margarita Rincon**:
**Data Collection and Processing**
<ul>
<li>Dataset: SARD – Search and Rescue</li>
<li>Source: Dataset From: **https://universe.roboflow.com/datasets-pdabr/sard-8xjhy**</li>
<li>Collect aerial imagery taken in natural, emergency-response scenarios.</li>
<li>Separate labeled images into “people” and “no_people” categories.</li>
<li>Remove duplicates, artifacts, and unclear samples.</li>
<li>Resize all images to 224×224 and normalize.</li>
<li>Apply Data Augmentation</li>
<li>Split dataset into 70% training, 15% validation, and 15% testing subsets.</li>
</ul>

### Model's Plan 

**Aung's Model**
<ul>
<li>Build CNN from scratch: 
<ul><li>(Conv3×3, 32) → ReLU → BN → MaxPool</li> 
       <li>(Conv3×3, 64) → ReLU → BN → MaxPool</li>
       <li>(Conv3×3, 128) → ReLU → BN → GlobalAvgPool → Dropout(0.3) →Linear(128,2) </li>
</ul>
<li>Loss function and Optimizer: Cross-Entropy; Adam.</li> 
<li>Augmentations: RandomHorizontalFlip, RandonVerticalFlip, RandomRotation, ColorJitter, RandomResizedCrop.</li>
<li> Training: 20 epochs and fine tuning the variable according to results.</li>
<li>Outputs: Accuracy, Prediction grid</li>
<li> Link to dataset: https://drive.google.com/file/d/1DqctU2uuiGeLeU-YNASPM8zLB2nIVOq9/view?usp=drive_link</li>
</ul><br>

**Margarita Rincon's Model: MobileNetV2**

<ul>
  <li><strong>Transfer Learning Model</strong></li>
  <li>Used pretrained <strong>MobileNetV2</strong>, replacing the final classifier layer to output 2 classes (people / no_people).</li>
  <li>Loss & Optimizer: <strong>CrossEntropyLoss</strong> with <strong>Adam</strong> optimizer (lr=1e-4, weight_decay=1e-4).</li>
  <li>Augmentations: <strong>RandomHorizontalFlip</strong>, <strong>RandomRotation(10)</strong>, <strong>ColorJitter(0.1, 0.1, 0.1)</strong>.</li>
  <li>Training: <strong>20–30 epochs</strong> (longer possible); monitored training/validation loss to detect overfitting.</li>
  <li>Evaluation: <strong>Accuracy</strong>, <strong>Precision/Recall</strong>, <strong>Confusion Matrix</strong>; tracked Train vs Test performance.</li>
  <li>Goal: Evaluate how well pretrained features help detect small, distant humans in drone images compared to a custom CNN.</li>
</ul>


### Project Timeline

#### Wk 1(10/27 - 11/1)
Data Curation, formatting, and splitting.

#### Wk 2 (11/2 - 11/8)
Data research and start Constructing both of the models. <br>
Deliverable: Data pre-processing notebook.

#### Wk 3 (11/9 - 11/15)
Refining and tuning the model.
### Wk 4(11/16 - 11/22)
Analysis and Visualization.

#### Wk5(11/23 - 11/29)
Finalize README and polishing the notebooks. Prepare for the final presentation.

#### Wk6(11/30 - 12/6)
12/2/2025 : Give a presentation in the class about the Project. 
Deliverable: 8 min deck and demo images.

#### Wk7( 12/8 - 12/11)
Optimize the models by listening to the feedback from the presentation.<br>
Deliverable: 
<ol>
<li>Github repository</li>
<li>2x Model notebooks.</li>
<li>2x Analysis and Visualization Notebooks</li>
<li>2x Data pre-processing Notebook</li>
</ol>

## .gitignore
This repository includes a standard Python `.gitignore` file to exclude cache files, model weights, large datasets, and system artifacts from version control.
You can view the full file here: [`.gitignore`](./.gitignore)

## License
This project is released under the **MIT License**, allowing free use, modification, and distribution with proper attribution.  
See the full license text here: [LICENSE](./LICENSE)

