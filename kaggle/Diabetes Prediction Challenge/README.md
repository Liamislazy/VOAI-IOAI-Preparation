# Multi-Model Ensemble with Hill Climbing for Diabetes Diagnosis

This repository contains an end-to-end machine learning pipeline designed for high-performance binary classification. It leverages a **10-Fold Stratified Cross-Validation Stacking Architecture** that blends state-of-the-art gradient boosting frameworks (XGBoost, LightGBM, CatBoost) with a deep Neural Network, utilizing an optimized Hill Climbing selection strategy for the final ensemble.

---

## 🚀 Strategy & Architecture Overview

The pipeline leverages a diverse set of models to ensure robustness, mitigate overfitting, and capture both tabular structural interactions and deep non-linear representations.

### 1. Robust Validation Strategy
* **10-Fold Stratified Cross-Validation:** Preserves the target class distribution across all folds, ensuring highly reliable Out-Of-Fold (OOF) evaluation metrics that closely align with the unseen test set performance.

### 2. Multi-Model Diversity (7 Models)
To maximize ensemble variance reduction, the pipeline incorporates highly diverse modeling paradigms:
* **XGBoost (Tuned):** Fine-tuned tree structures using the `lossguide` growth policy and direct categorical feature support enabled.
* **XGBoost DART:** Introduces dropout mechanisms into the tree-building process to prevent early tree dominance and reduce overfitting.
* **LightGBM (Robust):** Deep-leaf structure designed for rapid, scalable, and highly accurate gradient boosting on GPUs.
* **LightGBM Extra Trees:** Uses extremely randomized trees as base learners inside the boosting loop to inject algorithmic diversity.
* **CatBoost (Deep):** Maximizes performance using specialized symmetric tree splits and direct categorical feature mapping.
* **CatBoost (Shallow):** Focuses on shallow depth splits ($\text{depth}=3$) to capture global, low-interaction signals.
* **Keras Multi-Layer Perceptron (MLP):** A deep neural network using `Swish` activations, `BatchNormalization`, and `Dropout` to pick up complex structural weights missed by decision trees.

### 3. Ensembling via Hill Climbing
Instead of simple averaging or unconstrained linear regression, a **Forward Selection Hill Climbing Algorithm** iteratively optimizes the ensemble weights over 2,000 steps. This process maximizes the **ROC AUC** score on OOF predictions, strictly eliminating non-contributing models.

---

## 🛠️ Feature Engineering Pipeline

The code transforms raw inputs into highly predictive features via three primary channels:

### External Target & Frequency Encoding
Using an external baseline dataset (`diabetes_dataset.csv`), the pipeline maps historical cohort statistics onto the current dataset partitions:
* **`ext_mean_[col]`:** Conditional target means smoothed with a global mean backup fallback for unmapped levels.
* **`ext_freq_[col]`:** Relative frequency distributions of values across the reference cohort to capture representation density.

### Medical & Domain-Specific Interventions
* **Cardiovascular Metrics:** Calculates **Pulse Pressure** ($\text{systolic} - \text{diastolic}$) and **Mean Arterial Pressure (MAP)**.
* **Metabolic Indicators:** Generates **Insulin Resistance Proxies** ($\text{triglycerides} / \text{HDL}$), **Cholesterol Ratios**, and **Non-HDL Cholesterol**.
* **Adiposity Risk Metrics:** Combines **Visceral Fat Inductions** ($\text{BMI} \times \text{waist-to-hip ratio}$) and **Body Shape Risk Factors**.
* **Metabolic Syndrome Risk Score:** A discrete accumulator score tracking clinical thresholds (e.g., $\text{BMI} > 30$, $\text{Triglycerides} > 150$, $\text{Blood Pressure} > 130$).
* **Genetic Interactions:** Combines family history risks directly scaled with current biometric factors ($\text{age} + \text{BMI}$).
* **Lifestyle Biases:** Extracts **Sedentary Ratios** relative to active weekly physical movement.

### Structural Transforms
* **Combinatorial Pairs:** Computes all unique 2-way categorical string combinations (e.g., `gender_ethnicity_interact`) to grant gradient boosters instant access to high-order interactions.
* **Non-Linear Transforms:** Applies $\log(1+x)$ transformations to highly skewed columns (`triglycerides`, `bmi`) alongside quadratic feature squaring ($\text{feature}^2$).

---

## 💻 Tech Stack & Frameworks

* **Data Wrangling:** `Pandas`, `Numpy`, `Scikit-Learn`
* **Gradient Boosting:** `XGBoost`, `LightGBM`, `CatBoost`
* **Deep Learning:** `TensorFlow` / `Keras`
* **Hardware Acceleration:** Full GPU workflow routing for XGBoost (`gpu_hist`), LightGBM (`device='gpu'`), CatBoost (`task_type='GPU'`), and Keras memory management (`set_memory_growth`).

---
