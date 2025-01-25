Readme File
Assignment 4: Flower Image Classification with CNNs
Student Name: Adir Serruya
Submission Date: 2/2/2025
Institution: BGU

1. Description
This repository contains the implementation for Assignment 4, where we use two pre-trained CNN models, YOLOv5 and VGG19, to classify flower images into their respective categories. The task was performed on the Flowers102 dataset, containing 102 flower categories, and evaluated using PyTorch and Torchvision libraries.

2. Pretrained Models
YOLOv5:
A state-of-the-art object detection model adapted for classification.
Modified for multi-class flower classification with 102 categories.
VGG19:
A pre-trained image classification model.
Fine-tuned for flower classification by modifying the last fully connected layer.
3. Dataset
Flowers102 Dataset:
Link: Oxford Flowers Dataset
Total Images: 8,189
Categories: 102 flower types.
Data Split:
Training: 50%
Validation: 25%
Testing: 25%.
4. Key Features
Preprocessing:

Images resized to 224x224.
Normalization using ImageNet's mean and standard deviation.
Training Process:

Early stopping mechanism to prevent overfitting.
Optimizers: Adam with a learning rate of 0.005.
Loss Function: Cross-Entropy Loss.
Evaluation Metrics:

Accuracy, F1 Score, Precision, Recall.
Results:

YOLOv5: Test Accuracy: 93.02%
VGG19: Test Accuracy: 77.54%
5. Repository Contents
Code:

Python scripts for loading data, training models, and evaluating performance.
Includes custom dataset loaders and early stopping implementation.
Plots:

Accuracy and loss curves for YOLOv5 and VGG19.
PDF Report:

Contains detailed explanations of the models, preprocessing, training process, and results.


Links
GitHub Repository: [Link to Repository](https://github.com/Adirser/MLCourseUniv/tree/main/HW4%20-%20CNNs)
