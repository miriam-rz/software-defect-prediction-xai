# Software Defect Prediction with Explainable AI (XAI)

This repository contains the code developed for the thesis:

> **"Comparison of Explainable Artificial Intelligence Techniques in Software Defect Prediction"**
> Miriam Ramírez Zárate — Universidad Veracruzana, Faculty of Statistics and Informatics
> Bachelor's in Software Engineering, February 2026

## Overview

This project evaluates seven machine learning classifiers for software defect prediction on five NASA PROMISE datasets, and applies three post-hoc XAI techniques (LIME, SHAP, and BreakDown) to explain the predictions of the black-box models. Explanation quality is assessed using fidelity, stability, complexity, and interpretability metrics.

## Datasets

All datasets come from the NASA PROMISE repository and contain software metrics at the module level, with a binary target variable indicating the presence of defects.

| Dataset | Instances | Features |
|---------|-----------|----------|
| CM1     | 443       | 21       |
| JM1     | 8,839     | 21       |
| KC1     | 1,274     | 21       |
| KC2     | 389       | 21       |
| PC1     | 949       | 21       |

## Models

**Naturally interpretable:**
- Decision Tree
- Logistic Regression
- Naive Bayes (GaussianNB)

**Black-box:**
- Random Forest
- Gradient Boosting
- Support Vector Machine (SVM)
- Multilayer Perceptron (MLP)

## XAI Techniques

| Technique | Library | Explainer type |
|-----------|---------|----------------|
| LIME | `lime` | Local, model-agnostic |
| SHAP | `shap` | Local, TreeExplainer / KernelExplainer |
| BreakDown | `dalex` | Local, additive decomposition |

## Repository Structure

```
software-defect-prediction-xai/
├── datasets/
│   ├── CM1-T.csv
│   ├── JM1-T.csv
│   ├── KC1-T.csv
│   ├── KC2-T.csv
│   └── PC1-T.csv
├── notebooks/
│   ├── 01_dataset_preparation.ipynb
│   ├── 02_descriptive_analysis.ipynb
│   ├── 03_train_explainable_models.ipynb
│   ├── 04_train_black_box_models.ipynb
│   ├── 05_explain_lime.ipynb
│   ├── 06_explain_shap.ipynb
│   ├── 07_explain_breakdown.ipynb
│   └── 08_evaluate_explanations.ipynb
├── src/
│   └── utils.py
├── environment.yml
└── README.md
```

Notebooks should be run in order. Each one generates the inputs needed by the next.

## Setup

**1. Clone the repository**
```bash
git clone https://github.com/miriam-rz/software-defect-prediction-xai.git
cd software-defect-prediction-xai
```

**2. Create and activate the environment**
```bash
conda env create -f environment.yml
conda activate xai_env
```

**3. Launch JupyterLab**
```bash
jupyter lab
```

**4. Run the notebooks in order**, starting with `01_dataset_preparation.ipynb`.

The `datasets/processed/` and `results/` folders are generated automatically when the notebooks are executed.

## Evaluation Metrics

Model performance is evaluated using Accuracy, Precision, Recall, and F1-score, averaged across 5 stratified folds. SMOTE is applied inside each fold to address class imbalance.

Explanation quality is evaluated using:
- **Fidelity**: whether the explainer's prediction matches the model's prediction
- **Stability**: consistency of explanations across instances
- **Complexity**: proportion of features used in each explanation
- **Interpretability score**: average of fidelity and stability
