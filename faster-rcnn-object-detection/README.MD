# Faster R-CNN Object Detection from Scratch (TensorFlow)

## Overview

This project demonstrates a simplified implementation of the **Faster R-CNN object detection architecture** using **TensorFlow and Keras**.

The goal of this project is to understand the **complete object detection pipeline**, including feature extraction, region proposal generation, ROI feature extraction, and classification.

The implementation focuses on the **core concepts behind Faster R-CNN** rather than building a production-level detection system.

---

## Project Features

✔ CNN Backbone for feature extraction  
✔ Region Proposal Network (RPN)  
✔ ROI Align implementation  
✔ Detection head for classification and bounding box regression  
✔ Image testing using CIFAR-10 dataset  
✔ Video frame inference using OpenCV  
✔ Visualization of predicted bounding boxes  

---

## Architecture Pipeline

Image → Backbone CNN → Feature Map → Region Proposal Network → ROI Align → Detection Head → Class Prediction + Bounding Box

---

## Technologies Used

- Python
- TensorFlow
- Keras
- OpenCV
- NumPy
- Matplotlib

---

## Dataset

The project uses the **CIFAR-10 dataset** for demonstration.

Classes include:

- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

---

## Project Files

| File | Description |
|-----|-------------|
| `faster_rcnn_pipeline.py` | Full Faster R-CNN pipeline implementation |
| `cnn_padding_sliding_window.py` | Demonstrates CNN padding and sliding window concepts |

---

## How to Run

### Install Dependencies

```bash
pip install tensorflow numpy matplotlib opencv-python
```

### Run the Model

```bash
python faster_rcnn_pipeline.py
```

---

## Model Components Implemented

### 1. Backbone Network
A convolutional neural network extracts spatial feature maps from input images.

### 2. Region Proposal Network (RPN)
The RPN predicts potential object regions and bounding box coordinates.

### 3. ROI Align
ROI Align extracts fixed-size feature maps from proposed regions.

### 4. Detection Head
The detection head predicts:

- Object class
- Bounding box offsets

---

## Real World Applications

Object detection systems like Faster R-CNN are used in many real-world applications:

• Autonomous driving (detect cars, pedestrians)  
• Airport security monitoring  
• Surveillance systems  
• Medical image analysis  
• Smart retail analytics  
• Robotics and automation  

---

## What I Learned

Through this project I gained practical understanding of:

- CNN feature extraction
- Region Proposal Networks
- ROI Align operation
- Bounding box regression
- Object detection pipelines
- Deep learning model design
- Computer vision model deployment concepts

I also learned how modern object detection architectures combine **classification and localization in a single pipeline**.

---

## Future Improvements

Possible extensions for this project:

- Implement anchor box generation
- Train the RPN properly
- Use a larger dataset (COCO or Pascal VOC)
- Implement non-maximum suppression (NMS)
- Improve bounding box prediction
- Deploy model as an API

---

