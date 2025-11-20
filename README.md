## Description
This is the Final Project for CS_171/SJSU/Fall_2025 class, which explores how convolutional neural networks (CNNs) can distinguish between aerial drone images that contain people and those that do not. Using RGB imagery from the VisDrone and SAR Search & Rescue datasets, we train and evaluate models that classify each image into “people” or “no people.” The goal is to develop a lightweight and efficient vision model suitable for real-time aerial analysis.

## Project Title
**AerialWatch: Detection of Human presence from Drone Images**

## Authors
**Aung Aung** , Github: jaz4Aung <br>
**Margarita Rincon**, Github: Mago-RM

## Question and Research Topic
We want to observe if a Convolutional neural network can reliably identify the presence and absence of humans from aerial “RGB”  images taken by drones. <br>
Using urban aerial images of people, we will train and evaluate our models to detect pedestrians at different altitudes and crowd sizes. <br>
This research explores challenges of **small-object detection** caused by low resolution, obstructions, and different camera angles.<br>
We evaluate the robustness to scale, viewpoint, and scene diversity, and report accuracy, ROC-AUC, and calibration.<br>
We will analyse failure cases and discuss how we can effectively increase generalization of the models.


## Project Outline
### Data Collection Plan

**Aung**: 
<ul>
<li>Download VisDrone Images and gather images with visible pedestrians for  people and empty senses for no_people.<br> Link: https://github.com/VisDrone/VisDrone-Dataset</li>
<li> Clean the data by removing duplicates and blurry photos, photo size adjustments and normalize the data.</li>
<li>Create 70/15/15/ splits and save files accordingly. </li>
</ul>

**Margarita Rincon**:
**Data Collection and Processing**
<ul>
<li>Dataset: SARD – Search and Rescue on Kaggle</li>
<li>Collect aerial imagery taken in natural, emergency-response scenarios.</li>
<li>Separate labeled images into “people” and “no_people” categories.</li>
<li>Remove duplicates, artifacts, and unclear samples.</li>
<li>Resize all images to 224×224 and normalize between 0–1.</li>
<li>Data Augmentation</li>
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
<li> Training: 150 epochs and fine tuning the variable according to results.</li>
<li>Outputs: Accuracy, ROC-AUC</li>
</ul><br>

**Margarita Rincon 's Model**

<ul>
<li>**Transfer Learning Model**</li>
<li>Pretrained ResNet18 (or MobileNetV2) fine-tuned for binary classification.</li>
<li>Loss & Optimizer: CrossEntropyLoss + Adam (lr=1e-4, weight_decay=5e-4)</li>
<li>Augmentations: RandomRotation, RandomHorizontalFlip, RandomCrop</li>
<li>Training: 40–60 epochs with early stopping based on validation loss.</li>
<li>Evaluation Metrics: Accuracy, ROC-AUC, Precision/Recall, Confusion Matrix.</li>
<li>Goal: Compare performance between models to assess efficiency of pretrained features on smaller datasets.</li>
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

