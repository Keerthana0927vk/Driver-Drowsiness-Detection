# Driver Drowsiness Detection using Eye Closure and Yawning Analysis with Deep Learning

## Project Overview

Driver fatigue is one of the leading causes of road accidents worldwide. Drowsy drivers experience reduced alertness, slower reaction times, and poor decision-making abilities, increasing the risk of collisions. Traditional fatigue detection systems often rely on wearable sensors or vehicle behavior monitoring, which may be intrusive or unreliable.

This project presents a vision-based driver drowsiness detection system using Deep Learning and Computer Vision techniques. The system detects fatigue by analyzing physiological facial indicators such as eye closure and yawning behavior. A custom CNN model and a MobileNetV2 Transfer Learning model were developed and compared for performance.

---

## Problem Statement

Develop a non-intrusive driver monitoring system capable of detecting driver drowsiness using facial image classification.

The system classifies images into four categories:

* Open
* Closed
* no_yawn
* yawn

The predictions are further converted into three fatigue levels:

| Prediction | Fatigue Level  |
| ---------- | -------------- |
| Open       | Alert          |
| no_yawn    | Alert          |
| yawn       | Mild Fatigue   |
| Closed     | Severe Fatigue |

---

## Technologies Used

* Python
* TensorFlow
* Keras
* OpenCV
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn

---

## Project Workflow

### 1. Data Collection

The dataset contains facial images representing:

* Eyes Open
* Eyes Closed
* Yawn
* No Yawn

The dataset was organized into:

* Training Set
* Validation Set
* Test Set

---

### 2. Data Preprocessing

The following preprocessing techniques were applied:

* Image resizing to 224 × 224
* Pixel value normalization
* Data augmentation

  * Rotation
  * Zoom
  * Horizontal Flip

---

### 3. Custom CNN Model

A Convolutional Neural Network was developed using:

* Conv2D Layers
* MaxPooling Layers
* Dense Layers
* Dropout Layer

The CNN model was trained and evaluated on the facial image dataset.

---

### 4. MobileNetV2 Transfer Learning

To improve performance, MobileNetV2 pretrained on ImageNet was used.

Steps performed:

* Loaded pretrained MobileNetV2
* Removed top classification layer
* Added custom classification layers
* Frozen pretrained layers
* Trained classification head

---

## Model Evaluation

Evaluation metrics used:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

---

## Results

### Custom CNN Model

| Metric   | Value |
| -------- | ----- |
| Accuracy | 71%   |

The CNN model successfully classified eye and mouth states but showed limitations in generalization compared to transfer learning.

---

### MobileNetV2 Model

| Metric   | Value |
| -------- | ----- |
| Accuracy | 85%   |

MobileNetV2 significantly outperformed the custom CNN model and achieved better classification performance across most classes.

---

## Fatigue Classification Logic

The four-class predictions were converted into three fatigue stages:

### Alert

* Open
* no_yawn

### Mild Fatigue

* yawn

### Severe Fatigue

* Closed

This rule-based approach provides an interpretable fatigue assessment system.

---

## Driver Fatigue Progression Analysis

A fatigue progression curve was generated using sequential predictions.

The progression curve demonstrates how driver fatigue changes over time and identifies transitions between:

* Alert
* Mild Fatigue
* Severe Fatigue

This helps monitor fatigue trends during continuous driving sessions.

---

## Project Structure

Driver-Drowsiness-Detection/

├── dataset/

├── models/

│   ├── cnn_model.keras

│   └── mobilenet_model.keras

├── outputs/

│   ├── cnn_accuracy.png

│   ├── cnn_loss.png

│   ├── confusion_matrix.png

│   ├── mobilenet_accuracy.png

│   ├── mobilenet_loss.png

│   ├── mobilenet_confusion_matrix.png

│   └── fatigue_progression_curve.png

├── src/

│   ├── train_cnn.py

│   ├── train_mobilenet.py

│   ├── fatigue_logic.py

│   └── progression_curve.py

└── README.md

---

## Business Applications

* Road Safety Systems
* Fleet Management
* Driver Monitoring Systems
* Advanced Driver Assistance Systems (ADAS)
* Insurance Risk Assessment
* Smart Mobility Solutions


## Conclusion

This project successfully developed a deep learning-based driver drowsiness detection system using eye closure and yawning analysis. Both a Custom CNN and MobileNetV2 Transfer Learning model were implemented and compared. MobileNetV2 achieved superior performance with 85% accuracy and demonstrated strong capability in detecting fatigue-related facial states. The final fatigue classification logic and progression analysis provide a practical foundation for real-time driver monitoring applications.
