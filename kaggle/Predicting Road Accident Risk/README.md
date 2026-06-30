# Predicting Road Accident Risk - Multi-Booster Ensemble Pipeline

This repository contains an end-to-end machine learning solution designed to predict road accident risk using tabular data. The architecture relies on a **5-Fold Cross-Validation Stacking Framework** that combines three premier gradient boosting algorithms (XGBoost, LightGBM, and CatBoost) optimized via an inverse-performance weighted ensemble.

---

## 🚀 Strategy & Architecture Overview

The pipeline implements an automated workflow tailored for robust, high-capacity tabular forecasting on large-scale datasets (~517k training samples).
              ┌───────────────────────┐
              │    Raw Input Data     │
              └───────────┬───────────┘
                          │
                ┌─────────┴─────────┐
                ▼                   ▼
      ┌───────────────────┐┌───────────────────┐
      │Domain Interactions││Group Aggregations │
      └─────────┬─────────┘└─────────┬─────────┘
                └─────────┬──────────┘
                          │
                          ▼
                ┌───────────────────┐
                │ One-Hot Alignment │
                └─────────┬─────────┘
                          │
           ┌──────────────┼──────────────┐
           ▼              ▼              ▼
     ┌───────────┐  ┌───────────┐  ┌───────────┐
     │  XGBoost  │  │  LightGBM │  │  CatBoost │
     │  (Cuda)   │  │   (GPU)   │  │   (GPU)   │
     └─────┬─────┘  └─────┬─────┘  └─────┬─────┘
           └──────────────┼──────────────┘
                          │ (5-Fold OOF Predictions)
                          ▼
              ┌───────────────────────┐
              │ Inverse-RMSE Ensemble │
              └───────────┬───────────┘
                          ▼
              ┌───────────────────────┐
              │ Final Bound-Clipped   │
              │      Submission       │
              └───────────────────────┘
### 1. Cross-Validation Framework
* **5-Fold Cross Validation (`KFold`):** The data is split into five random partitions to generate reliable Out-Of-Fold (OOF) validation predictions. This safeguards against dataset leakage and guarantees an accurate proxy metric for the unseen test leaderboard.

### 2. Heterogeneous Gradient Boosting Mix
To balance bias and maximize variance reduction, the pipeline utilizes three complementary boosting structures, fully optimized for GPU acceleration:
* **XGBoost Regressor:** Configured with `tree_method: 'hist'` and `device: 'cuda'` using structural `reg:squarederror` optimization and an early stopping cushion of 250 rounds.
* **LightGBM Regressor:** Implements an $L_1$ absolute loss variation objective (`regression_l1`) to provide extreme robust protection against dataset outliers.
* **CatBoost Regressor:** Processes structural symmetric decision paths directly on the GPU using fine-grained learning controls.

### 3. Automated Inverse-Performance Ensemble
Instead of simple arithmetic averaging, weights are assigned programmatically based on the relative performance of each model. Weights are calculated via inverse OOF RMSE scores:

$$w_i = \frac{\frac{1}{\text{RMSE}_i}}{\sum_{j} \frac{1}{\text{RMSE}_j}}$$

This systematically increases the influence of the most accurate model while maintaining diversity from the other frameworks.

---

## 🛠️ Feature Engineering Pipeline

The feature engineering module introduces advanced structural signals by analyzing physical road conditions and metadata:

### Kinematic & Geometric Interactions
* **Curvature-to-Speed Ratios:** Computes risk indices highlighting sharp turns at high speed bounds (`curvature / speed_limit`).
* **Lane-Density Speeds:** Evaluates traffic velocity constraints against lane volumes (`num_lanes / speed_limit`).
* **Polynomial Complexities:** Computes quadratic feature squares (`curvature_sq`, `speed_limit_sq`) to capture non-linear degradation risks.

### Group-By Contextual Aggregations
Using structural road categories (`road_type`, `lighting`, `weather`, `time_of_day`), the pipeline groups geometric factors to find anomalies:
* **Conditional Statistics:** Extracts category-specific rolling `mean` and standard deviation (`std`) matrices for speed and curves.
* **Deviation Deltas:** Calculates the structural distance of the current segment from its categorical baseline (`num_col - mean`).

### Combinatorial Strings & Boolean Unions
* **Environmental Cross-Features:** Creates joint strings combining environmental states (e.g., `lighting_weather` and `road_type_lighting`) to catch high-risk situational overlaps (e.g., Nighttime + Heavy Rain).
* **Logical Indicators:** Flags intersection logic for concurrent risks, such as `holiday_and_school_season` and `signs_and_public_road`.

---

## 💻 Tech Stack & Optimizations

* **Data Wrangling:** `Pandas`, `Numpy`, `Scikit-Learn`
* **GPU Computing Stack:** Native `CUDA` backend pipelines for parallelized gradient evaluation.
* **Pipeline Alignment:** Features a clean post-encoding column check that handles dummy alignment, ensuring identical feature structural shape between training sets and inference arrays.

---

## 📈 Model Performance & Output Logs

Executing the pipeline produces balanced performance profiles across all sub-frameworks:

```bash
Training XGBoost...
XGBoost OOF RMSE: 0.056143

Training LightGBM...
LightGBM OOF RMSE: 0.056392

Training CatBoost...
CatBoost OOF RMSE: 0.056154

============================================================
Calculating weights based on model performance...
Best Weights -> XGB: 0.334, LGB: 0.332, CAT: 0.334
Final Weighted Ensemble OOF RMSE: 0.056126
============================================================

Creating submission file...
✅ Submission file 'submission.csv' created successfully (Bound-clipped [0, 1]).
