# Credit Card Fraud Detection with Gaussian Naive Bayes

## Overview

This project applies **Gaussian Naive Bayes (GaussianNB)** to credit card fraud detection using the Credit Card Fraud Detection dataset. The notebook focuses on understanding how a probabilistic classification model performs on a highly imbalanced dataset and how preprocessing techniques such as feature scaling and SMOTE affect its performance.

A **Logistic Regression** model is also included as a secondary baseline for comparison.

## Problem Statement

Credit card fraud detection is a binary classification problem where transactions are classified as:

- `0` — Legitimate transaction
- `1` — Fraudulent transaction

The main challenge is severe class imbalance. Fraudulent transactions represent only a very small portion of the dataset, making accuracy alone an unreliable evaluation metric.

Therefore, this project focuses on:

- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Confusion-matrix-based evaluation

## Dataset

The project uses the **Credit Card Fraud Detection** dataset.

### Dataset Characteristics

- **284,807 transactions**
- **31 columns**
- **284,315 legitimate transactions**
- **492 fraudulent transactions**

### Features

- `Time` — elapsed time between transactions
- `V1`–`V28` — anonymized PCA-transformed numerical features
- `Amount` — transaction amount
- `Class` — target variable

The `Class` column is the prediction target. `Time` is excluded from the modeling features in this project.

## Machine Learning Workflow

The notebook follows this workflow:

1. Load the dataset
2. Inspect the dataset
3. Check for missing values
4. Analyze class distribution
5. Separate features and target
6. Perform a stratified train/test split
7. Train a baseline Gaussian Naive Bayes model
8. Evaluate training and testing performance
9. Test GaussianNB with feature scaling
10. Test GaussianNB with SMOTE
11. Test GaussianNB with SMOTE and feature scaling
12. Compare with Logistic Regression
13. Evaluate ROC-AUC
14. Evaluate PR-AUC
15. Analyze the results

## Experiments

### 1. Gaussian Naive Bayes — Baseline

A standard `GaussianNB` classifier is trained on the original training data. This establishes a baseline before applying additional preprocessing techniques.

### 2. Gaussian Naive Bayes + Scaling

`StandardScaler` is used to standardize the features.

The scaler is fitted only on the training data and then applied to the test data to prevent data leakage.

This experiment investigates whether feature standardization changes GaussianNB performance.

### 3. Gaussian Naive Bayes + SMOTE

Because the fraud class is highly underrepresented, **SMOTE (Synthetic Minority Over-sampling Technique)** is applied to the training data.

The test set is kept unchanged so that evaluation reflects the original class distribution.

### 4. Gaussian Naive Bayes + SMOTE + Scaling

This experiment combines feature standardization and SMOTE.

The training data is scaled and then resampled using SMOTE. The test data is transformed using the scaler fitted on the training data and is not resampled.

### 5. Logistic Regression Comparison

Logistic Regression is included as a secondary baseline to provide context for the GaussianNB results.

Scaled and unscaled experiments use matching training and testing representations to maintain a valid comparison.

## Evaluation Metrics

### Accuracy

Accuracy measures the proportion of correctly classified transactions.

Because the dataset is extremely imbalanced, accuracy is **not treated as the primary metric**.

### Precision

Precision measures how many transactions predicted as fraud are actually fraudulent.

### Recall

Recall measures how many of the actual fraudulent transactions are successfully detected.

### F1-Score

F1-score combines precision and recall into a single metric and is useful when both false positives and false negatives matter.

### ROC-AUC

ROC-AUC measures the model's ability to distinguish between legitimate and fraudulent transactions across different classification thresholds.

### PR-AUC

Precision-Recall AUC is particularly useful for this project because fraud is a very rare positive class. It summarizes the precision-recall trade-off across different thresholds.

## Handling Class Imbalance

The dataset contains far more legitimate transactions than fraudulent transactions.

A model can therefore achieve very high accuracy while performing poorly on the fraud class.

To investigate this problem, the project uses **SMOTE** to create synthetic minority-class training examples.

> **Important:** SMOTE is applied only to the training data. The test set remains untouched.

This prevents test-set information from leaking into the training process.

## Key Learning Outcomes

This project demonstrates:

- Application of Gaussian Naive Bayes to binary classification
- Handling of highly imbalanced datasets
- Why accuracy can be misleading for fraud detection
- Stratified train/test splitting
- Correct use of feature scaling without data leakage
- Use of SMOTE for minority-class oversampling
- Evaluation using precision, recall, F1-score, ROC-AUC, and PR-AUC
- Comparison of GaussianNB with Logistic Regression
- Understanding the effect of preprocessing choices on model performance

## Project Structure

```text
credit-card-fraud-gaussian-nb/
│
├── ml_naive_bayes.ipynb
├── README.md

```

> The dataset does not need to be committed to GitHub if repository size or dataset licensing is a concern. The notebook can download the dataset using `kagglehub`.

## Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn kagglehub jupyter
```

## Running the Project

1. Clone the repository.
2. Install the required dependencies.
3. Open the notebook using Jupyter Notebook or JupyterLab.
4. Run the cells from top to bottom.

The notebook uses `kagglehub` to download the Credit Card Fraud Detection dataset.

## Conclusion

This project provides a practical introduction to **Gaussian Naive Bayes for imbalanced binary classification**.

The experiments demonstrate that preprocessing techniques such as feature scaling and SMOTE should be evaluated based on their actual effect rather than assumed to improve a model. For fraud detection, accuracy alone is not sufficient; precision, recall, F1-score, ROC-AUC, and PR-AUC provide a more meaningful assessment.

Gaussian Naive Bayes serves as a useful probabilistic baseline and provides a foundation for exploring more advanced classification algorithms in future projects.

## Future Improvements

Possible extensions include:

- Classification threshold optimization
- Cross-validation
- Additional classification algorithms
- Hyperparameter experimentation
- More detailed confusion-matrix analysis
- Cost-sensitive learning
- Probability calibration
- Model interpretability
- Ensemble and boosting methods

## Author

Machine Learning project focused on learning and evaluating **Gaussian Naive Bayes for credit card fraud detection**.
