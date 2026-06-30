# Predicting Loan Payback - Heterogeneous Stacking Ensemble

This repository contains a high-performance machine learning pipeline built to predict loan payback probabilities using tabular financial records. The solution implements a 7-Fold Stratified Stacking Architecture combining three premier gradient boosting libraries (XGBoost, LightGBM, CatBoost) with a deep Keras Neural Network, meta-blended via Ridge Regression.

---

## Strategy and Architecture Overview

The pipeline handles large-scale data modeling using a two-stage heterogeneous stacking framework optimized for absolute variance reduction and high classification boundaries.

### 1. Processing Flow
* Raw Data -> Interaction Features + External Cohort Mapping
* Pipeline Transforms -> Continuous Scaling / One-Hot Coding / Category Ordinal Assignments
* Base Modeling -> Parallel execution of XGBoost, LightGBM, CatBoost, and a Keras Neural Network
* Ensemble Blending -> 7-Fold Stratified OOF Array Generation -> Ridge Regression Meta-Model -> Threshold Calibration

### 2. Stratified Validation Strategy
* 7-Fold Stratified Cross-Validation (StratifiedKFold): Maintains perfect representation of the target binary balance across all execution blocks, generating bulletproof Out-Of-Fold (OOF) metadata arrays entirely free of target leakage.

### 3. Multi-Paradigm Base Layer (4 Models)
* XGBoost Classifier: Deep tree growth capped at max_depth=5, incorporating robust regularizations (reg_alpha=0.05, reg_lambda=1.0) on CUDA-accelerated histogram evaluation.
* LightGBM Classifier: Configured with 50 leaves per tree, optimizing direct log-loss objectives under GPU acceleration structures.
* CatBoost Classifier: Employs symmetric tree splits with high structural noise suppression (bagging_temperature=0.5) running natively on the GPU.
* Keras Multi-Layer Perceptron (MLP): A deep neural architecture mapping 4 dense structural layers (512 -> 256 -> 128 -> 64) layered with BatchNormalization, ReLU activations, and Dropout components to capture non-linear weight boundaries missed by decision trees.

### 4. Ridge Stacking and Threshold Optimization
Instead of a simple blend, a Ridge Regression Meta-Model trains directly on the stacked OOF probability vectors. Post-training, a granular threshold search engine scans parameters from 0.10 to 0.89 to systematically maximize the classification accuracy boundary.

---

## Advanced Feature Engineering

The feature module introduces sophisticated economic vectors derived from domain-specific metrics and cohort indicators:

### External Cohort Statistics (Leakage-Free Mapping)
Utilizing a separate, independent external reference dataset (loan_dataset_20000.csv), the script builds global historical lookup tables without leaking information from the competition target:
* orig_mean_[col]: Matches historical loan payback risk margins onto active tabular features.
* orig_count_[col]: Computes the overall frequency density within the baseline cohort to map representation strength.

### Financial Risk and Liquidity Metrics
* Debt and Affordability Vectors: Calculates total active debt obligations, remaining disposable income margins, and strict affordability metrics (available_income / loan_amount).
* Velocity of Capital: Extrapolates estimated monthly installment payments alongside overall payment-to-income liquidity proportions.
* Synthetic Credit Risk Score: Compiles clinical weighted risk indices combining credit metrics, interest penalty fields, and debt-to-income scores into a single structural feature.

### Structural Combinations and Transforms
* Hierarchical Credit Sorting: Deconstructs categorical credit risk labels into hierarchical integers, combining rank classifications with localized subgroup tiers (grade_rank * 10 + grade_number).
* Group-by Contextual Mappings: Group-aggregates numerical features over cross-categorical combinations inside the training folds to evaluate cross-sectional anomalies (cat_loan_mean, cat_credit_mean).
* Skew Realignment: Applies smooth non-linear transformations using log(1+x) parameters to stabilize high-variance financial inputs.

---

## Tech Stack and Framework Accelerations

* Core Engine: Pandas, Numpy, Scikit-Learn
* Deep Learning: TensorFlow / Keras sequential execution blocks.
* Pipeline Infrastructure: ColumnTransformer providing distinct paths for continuous metrics (Standard Scaling), categorical objects (One-Hot), and specialized credit structures (Categorical Ordinal constraints).
* Hardware Profile: Multi-GPU parallel training routing across native CUDA pipelines, LightGBM OpenCL builders, and TensorFlow backend runtimes.

---

## Pipeline Output Execution Logs

When executed, the script maps the feature transformations, sweeps through the 7 validation folds, and logs the finalized classification profiles:

```bash
Loading Data...
Train shape: (593994, 13)
Test shape: (254569, 12)
Original (External) shape: (20000, 22)

Applying Advanced Feature Engineering...
Feature Engineering Complete.
Total Features: 59 | Numerical: 52 | Categorical: 7

--- Starting 7-Fold CV for 4 Models ---
--- Fold 1/7 ---
Training XGB...
Training LGBM...
Training CatBoost...
Training NN...
...

--- Individual Model Accuracy ---
XGB: 0.90840
LGB: 0.90835
CAT: 0.90817
NN : 0.90654

--- Training Meta Model (Ridge) ---
Best OOF Accuracy: 0.908442 at threshold 0.50
File saved: submission_ridge_4stack_advanced_fe_acc_0_9084.csv
