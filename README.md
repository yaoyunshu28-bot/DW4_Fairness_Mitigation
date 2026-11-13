# Evaluating Reweighing

This repository contains the Python Notebook for Task 2 of the Data Wrangling assignment.

The notebook implements **Reweighing** from Calders, Kamiran and Pechenizkiy (2009), *Building Classifiers with Independency Constraints*.

Reweighing is implemented directly using NumPy. It calculates training weights and uses them to sample training records with replacement. The original labels are kept.

## Files

- `DW4_Task2_Reweighing.ipynb`: code, results and explanations.
- `data/`: created automatically when the notebook downloads the datasets.

## Datasets

| Dataset | Records used | Group 0 | Group 1 | Positive label |
|---|---:|---|---|---|
| Adult | 48,842 | Female | Male | Income above $50,000 |
| German Credit | 1,000 | Female | Male | Good credit risk |
| COMPAS | 5,278 | African-American | Caucasian | No recorded recidivism within two years |

Adult and German Credit come from UCI. COMPAS comes from ProPublica. The notebook applies the stated COMPAS filters before analysis.

## Mitigation Pipeline

The notebook performs these steps for each dataset:

1. Load and prepare the data.
2. Calculate fairness scores from the original labels before training.
3. Split the data into 70% training and 30% test data, stratified by group and label together.
4. Fit the scaler and encoder on the original training features.
5. Calculate Reweighing weights from the training data and sample training records with replacement.
6. Train two logistic regression models: one using the original training records and one using the Reweighing sample.
7. Compare both models on the same held-out test records.

The sensitive attribute is used to calculate weights and fairness scores. It is excluded from the classifier inputs.

## Parameter Settings

| Setting | Value |
|---|---|
| Random seed | 42 |
| Train/test split | 70% / 30% |
| Classifier | Logistic Regression |
| C | 1.0 |
| Solver | liblinear |
| Maximum iterations | 2,000 |
| Positive prediction threshold | 0.5 |
| Numerical features | StandardScaler |
| Categorical features | OneHotEncoder |
| Neighbours for consistency | 5, with self-matches allowed |

Nearest-neighbour search is used only to calculate consistency. It is not used as a classification model.

## Evaluation Measures

Let `r0` and `r1` be the positive rates in groups 0 and 1.

- **DP_Diff = r0 − r1:** closer to 0 means a smaller group gap.
- **DIR = r0 / r1:** closer to 1 means more equal positive rates.
- **Consistency:** higher values mean that nearby records receive more similar labels or predictions.
- **Accuracy:** the share of test predictions that match the original test labels.

The first output table reports fairness scores for the original labels. The second reports fairness scores and accuracy for both test models.

## How to Run

Clone or download this repository. Install the required packages:

```bash
python -m pip install numpy pandas scipy "scikit-learn>=1.2" ipython jupyterlab
```

Start JupyterLab:

```bash
jupyter lab
```

Open `DW4_Task2_Reweighing.ipynb` and select **Run All**. You can also open the notebook in VS Code with Jupyter support.

The notebook automatically downloads missing datasets into the local `data/` folder. No manual data download is needed.

The package versions used for the reported results are listed in the notebook.

## Sources

- [Calders, Kamiran and Pechenizkiy (2009)](https://doi.org/10.1109/ICDMW.2009.83)
- [UCI Adult](https://archive.ics.uci.edu/dataset/2/adult)
- [UCI German Credit](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)
- [ProPublica COMPAS data](https://github.com/propublica/compas-analysis)
