# Breast Cancer Ultrasound Classification using EfficientNetB0

## Overview

This project implements a deep learning model for **breast cancer ultrasound image classification** using **EfficientNetB0** and TensorFlow.

The model classifies ultrasound images into three categories:

- Benign
- Malignant
- Normal

The goal of this project is to demonstrate how **deep learning and transfer learning can assist in medical image diagnosis**.

The project also includes **model evaluation techniques and explainable AI using Grad-CAM** to understand how the model makes decisions.

---

## Project Objectives

The main objectives of this project are:

- Build a deep learning model for medical image classification
- Apply **transfer learning using EfficientNetB0**
- Handle **class imbalance in medical datasets**
- Evaluate model performance using multiple metrics
- Visualize model attention using **Grad-CAM**

---

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- OpenCV
- Scikit-learn
- Seaborn

These tools are used for deep learning model development, evaluation, and visualization.

---

## Dataset

The model is trained on a **breast ultrasound image dataset** containing three classes:

| Class | Description |
|------|-------------|
| Benign | Non-cancerous tumor |
| Malignant | Cancerous tumor |
| Normal | Healthy tissue |

Ultrasound imaging is commonly used for **early detection of breast cancer**.

---

## Model Architecture

The model uses **transfer learning with EfficientNetB0**.

Architecture pipeline:
Input Image (224x224)
        │
        ▼
EfficientNetB0 (Pretrained Feature Extractor)
        │
        ▼
Dense Layer
Batch Normalization
Dropout
        │
        ▼
Softmax Output Layer


---

## Data Preprocessing

The project uses **ImageDataGenerator** for data preprocessing and augmentation.

Techniques applied:

- Image normalization
- Random rotations
- Horizontal flipping
- Zoom augmentation
- Width and height shifting

These techniques help improve **model generalization**.

---

## Handling Class Imbalance

Medical datasets often contain **imbalanced classes**.

This project uses **class weights** computed using Scikit-learn to ensure balanced learning during training.

---

## Training Strategy

Training is performed in **two phases**:

### Phase 1 – Feature Extraction

- EfficientNetB0 layers are frozen
- Only the classifier layers are trained

### Phase 2 – Fine-Tuning

- Last layers of EfficientNet are unfrozen
- Model adapts to medical image features

This approach improves **model performance and stability**.

---

## Model Evaluation

Multiple evaluation metrics are used to measure performance:

- Accuracy
- Precision
- Recall
- AUC Score
- Confusion Matrix
- Classification Report

These metrics help evaluate the model’s **diagnostic reliability**.

---

## ROC Curve and AUC

ROC curves are used to evaluate how well the model distinguishes between different tumor classes.

A higher **AUC score** indicates better diagnostic performance.

---

## Explainable AI – Grad-CAM

To improve model interpretability, **Grad-CAM visualization** is used.

Grad-CAM highlights the regions of the ultrasound image that influenced the model’s prediction.

This is important for **medical AI systems**, where explainability is critical.

---

## Real World Applications

AI-based medical image analysis systems can support doctors in:

- Early breast cancer detection
- Tumor classification
- Radiology assistance systems
- Clinical decision support
- Automated diagnostic tools

Such systems can improve **diagnosis speed and accuracy**.

---

## What I Learned

Through this project I learned:

- Transfer learning with EfficientNet
- Medical image classification techniques
- Handling imbalanced datasets
- Model evaluation using ROC and AUC
- Confusion matrix analysis
- Explainable AI using Grad-CAM
- Deep learning workflow for healthcare applications

This project helped me understand how **AI can assist healthcare professionals in medical diagnosis**.

---

## Future Improvements

Possible improvements for this project include:

- Training with larger medical datasets
- Implementing advanced CNN architectures
- Hyperparameter optimization
- Deploying the model as a medical AI application
- Integrating with clinical decision support systems

---
