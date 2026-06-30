# MNIST Digit Recognition - 5-Fold CNN Ensemble with Albumentations

This repository contains a high-performance computer vision pipeline designed for handwritten digit classification. It implements a **5-Fold Cross-Validated Convolutional Neural Network (CNN) Ensemble** with dynamic image augmentation and learning rate schedules to achieve exceptional validation accuracy.

---

## Strategy and Architecture Overview

The pipeline leverages a robust deep-learning infrastructure to classify image matrices cleanly while protecting the architecture against model overfitting.

### 1. Processing Flow
* Raw Pixels -> Reshaped Tensor Formatting (28x28x1)
* Albumentations Pipeline -> Spatial Rotations / Scale Transforms
* Base CNN -> High-Capacity Convolution Extraction & Max-Pooling
* Ensemble Validation -> 5-Fold Validation Splits -> Stratified Prediction Averaging

### 2. Validation Framework
* 5-Fold Cross-Validation: Splits the entire image training corpus into five discrete subsets. Each partition acts as a rigorous validation testbed exactly once, providing cross-validated Out-Of-Fold (OOF) checkpoints to prevent data leakage.

### 3. Convolutional Neural Network Architecture
The framework initializes a Deep Keras Sequential model utilizing customized layer blocks:
* Feature Extraction: Hierarchical Convolutional layers (`Conv2D`) backed by localized pooling matrices (`MaxPooling2D`).
* Regularization: Interleaved Spatial Dropout configurations and Normalization boundaries to penalize co-adaptation.
* Classifier Head: Fully connected Dense layers transitioning down to a softmax activation layer mapping 10 distinct digit probabilities.

### 4. Dynamic Optimization Schedule
* Learning Rate Adjustments (ReduceLROnPlateau): Monitors validation losses in real-time. If localized metrics stall for more than 3 consecutive epochs, the scheduler automatically scales down the learning rate by a factor of 0.5 to gracefully find deeper global minima.

---

## Advanced Image Data Augmentation

To ensure morphological robustness against variations in handwriting styles, the processing loop infuses randomized alterations via the `Albumentations` engine during training steps:
* ShiftScaleRotate: Randomly shifts the text coordinate center, applies subtle zoom magnifications, and rotates baseline geometry.
* Value Fallbacks: Preserves strict boundaries on image edges using explicit zero-fill padding to maintain uniform matrix canvas sizes.

---

## Tech Stack and Framework Accelerations

* Core Framework: TensorFlow / Keras Sequential runtime engines
* Preprocessing Vector: Albumentations framework for parallel geometric data pipelines
* Data Handling: Numpy, Pandas, Scikit-Learn
* Hardware Acceleration: Fully mapped onto native GPU platforms (e.g., Tesla configurations) with optimized XLA compilers activated

---

## Pipeline Output Execution Logs

When executed, the system tracks fold generation, displays current training adjustments, and updates metrics smoothly across the training cycles:

```bash
Loading data...
Created device /job:localhost/replica:0/task:0/device:GPU:0 with 15511 MB memory: -> Tesla P100

========== FOLD 1/5 ==========
Epoch 1/15
525/525 [====================] - 14s 16ms/step - accuracy: 0.8194 - loss: 0.5928 - val_accuracy: 0.9650 - val_loss: 0.1136 - learning_rate: 0.0010
Epoch 2/15
525/525 [====================] - 8s 15ms/step - accuracy: 0.9539 - loss: 0.1546 - val_accuracy: 0.9787 - val_loss: 0.0670 - learning_rate: 0.0010
...
Epoch 10/15
Epoch 10: ReduceLROnPlateau reducing learning rate to 0.0005.
525/525 [====================] - 7s 14ms/step - accuracy: 0.9826 - loss: 0.0565 - val_accuracy: 0.9905 - val_loss: 0.0365 - learning_rate: 0.0010
...
Epoch 15/15
525/525 [====================] - 7s 14ms/step - accuracy: 0.9902 - loss: 0.0335 - val_accuracy: 0.9937 - val_loss: 0.0315 - learning_rate: 0.0005
Adding Fold 1 predictions to ensemble...

========== FOLD 2/5 ==========
...
