# Bias Mitigation using Reweighing (Data Wrangling)

This repository contains the implementation for Task 2 of the Data Wrangling assignment, evaluating the effectiveness of a bias-mitigation algorithm across three datasets.

## Algorithm
- **Method**: Reweighing (Pre-processing)
- **Library**: IBM AIF360

## Datasets Used
The pipeline automatically downloads and processes the following datasets:
1. **Adult Income** (UCI)
2. **German Credit** (UCI)
3. **COMPAS** (ProPublica)

## Implementation Details
- **Sensitive Attributes**: Sex (Adult, German), Race (COMPAS)
- **Classifiers Evaluated**: Logistic Regression, K-Nearest Neighbors (KNN)
- **Fairness Metrics Evaluated**: Demographic Parity Difference (DP_Diff), Disparate Impact Ratio (DIR), Consistency.
- **Predictive Metric**: Accuracy.

## How to Run
1. Clone this repository.
2. Install required libraries: `pip install aif360 scikit-learn pandas`
3. Open `DW4_Task2_Reweighing.ipynb`.
4. Run all cells. The code will automatically download the datasets into a local `data/` folder and execute the complete bias mitigation pipeline.
