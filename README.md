
# CS171_Final-Project

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

## Data Collection Plan

**Aung**: 
<ul>
<li>Download VisDrone Images and gather images with visible pedestrians for  people and empty senses for no_people.<br> Link: https://github.com/VisDrone/VisDrone-Dataset?utm_source=chatgpt.com</li>
<li> Clean the data by removing duplicates and blurry photos, photo size adjustments and normalize the data.</li>
<li>Create 70/15/15/ splits and save files accordingly. </li>
</ul>


**Margarita Rincon**: 
<ul>
<li>  </li>
<li>  </li>
<li>  </li>

</ul>


## Model's Plan 

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
<li>  </li>
<li>  </li>
<li>  </li>

</ul>

## Project Timeline

### Wk 1(10/27 - 11/1)
Data Curation, formatting, and splitting.

### Wk 2 (11/2 - 11/8)
Data research and start Constructing both of the models. <br>
Deliverable: Data pre-processing notebook.

### Wk 3 (11/9 - 11/15)
Refining and tuning the model.
### Wk 4(11/16 - 11/22)
Analysis and Visualization.

### Wk5(11/23 - 11/29)
Finalize README and polishing the notebooks. Prepare for the final presentation.

### Wk6(11/30 - 12/6)
12/2/2025 : Give a presentation in the class about the Project. 
Deliverable: 8 min deck and demo images.

### Wk7( 12/8 - 12/11)
Optimize the models by listening to the feedback from the presentation.<br>
Deliverable: 
<ol>
<li>Github repository</li>
<li>2x Model notebooks.</li>
<li>2x Analysis and Visualization Notebooks</li>
<li>2x Data pre-processing Notebook</li>
</ol>

