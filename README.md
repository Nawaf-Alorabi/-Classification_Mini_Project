# 🌲 Random Forest Classifier - Loan Approval Prediction

> Mini Project | Classification Algorithms Exploration  
> Dataset: [Loan Approval Dataset - Kaggle](https://www.kaggle.com/datasets/anishdevedward/loan-approval-dataset)

---

## 📌 Project Overview

This project explores the **Random Forest Classifier** algorithm and applies it to a real-world loan approval dataset. The goal is to predict whether a loan application will be **approved or rejected** based on applicant financial features.

The project covers the full machine learning pipeline:
- Algorithm explanation
- Exploratory Data Analysis (EDA) & Visualization
- Data Preprocessing
- Model Training
- Model Evaluation

---

## 🧠 Algorithm - Random Forest

Random Forest is an **ensemble learning** method that builds multiple Decision Trees during training and merges their results to produce a more accurate and stable prediction.

**How it works:**
1. Randomly samples subsets of the training data (**Bootstrap Sampling**)
2. Builds a Decision Tree on each subset
3. Each tree uses a **random subset of features** at every split (Feature Subsetting)
4. Final prediction is made by **majority vote** across all trees

**When to use it:**
- When data has many features and complex relationships
- When you need high accuracy with low risk of overfitting
- When interpretability of individual trees is less important than overall performance

**Advantages:**
- Handles missing values and outliers well
- Resistant to overfitting compared to a single Decision Tree
- Provides feature importance rankings
- Works well on both small and large datasets

**Limitations:**
- Slower to train than a single Decision Tree
- Less interpretable (black-box model)
- Requires more memory

---

## 📂 Dataset

| Property | Value |
|---|---|
| **Source** | Kaggle - `anishdevedward/loan-approval-dataset` |
| **File** | `loan_approval.csv` |
| **Rows** | 2,000 |
| **Columns** | 8 |
| **Target** | `loan_approved` (True / False) |

### Features

| Column | Type | Description |
|---|---|---|
| `name` | String | Applicant name (dropped in preprocessing) |
| `city` | String | Applicant city (dropped in preprocessing) |
| `income` | Integer | Annual income |
| `credit_score` | Integer | Credit score (300–850) |
| `loan_amount` | Integer | Requested loan amount |
| `years_employed` | Integer | Years at current employer |
| `points` | Float | Internal scoring points (10–100) |
| `loan_approved` | Boolean | ✅ **Target variable** |

---

## 📊 Exploratory Data Analysis (EDA)

The EDA section includes the following visualizations:

| # | Visualization | Purpose |
|---|---|---|
| 1 | Count plot + Pie chart | Class distribution of `loan_approved` |
| 2 | Histograms | Distribution of each numeric feature |
| 3 | Overlapping histograms | Feature distributions split by approval status |
| 4 | Box plots | Spread and outliers per feature by class |
| 5 | Correlation heatmap | Relationships between all features and target |
| 6 | Scatter plots | Credit Score vs Income / Points vs Credit Score |
| 7 | Missing values bar chart | Data completeness check |
| 8 | Grouped bar chart | Normalized mean comparison: Approved vs Rejected |

**Key EDA Findings:**
- `credit_score` and `points` show the strongest separation between approved and rejected applicants
- No missing values - dataset is clean
- `income` and `loan_amount` show less predictive power individually
- The dataset may be imbalanced - check the count plot output

---

## ⚙️ Data Preprocessing

Steps applied before model training:

1. **Drop irrelevant columns** - `name` and `city` are removed (not predictive)
2. **Encode target variable** - `loan_approved` converted from boolean to integer (1/0)
3. **No missing value imputation needed** - dataset is already clean
4. **Feature scaling** - applied using `StandardScaler` (important for distance-based comparisons)
5. **Train/Test split** - 80% training, 20% testing

---

## 🤖 Model Implementation

Two implementations are included in the notebook:

### ✅ Scikit-learn (Standard)
```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

### 🔧 Custom PyTorch Implementation
A manual implementation of the Random Forest algorithm built from scratch using PyTorch tensors, including:
- Custom `DecisionTree` class with Gini Impurity splitting
- Bootstrap sampling
- Random feature subsetting (√n features per split)
- Majority voting across trees

---

## 📈 Model Evaluation

The model is evaluated using:

| Metric | Description |
|---|---|
| **Accuracy** | Overall correct predictions |
| **Precision** | Of predicted approvals, how many were actually approved |
| **Recall** | Of actual approvals, how many were correctly predicted |
| **F1 Score** | Harmonic mean of Precision and Recall |
| **Confusion Matrix** | Visual breakdown of TP, TN, FP, FN |

```python
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, y_pred))
```

---

## 📁 Project Structure

```
📦 Classification_Mini_Project
 ┣ 📓 test.ipynb              ← Main Jupyter Notebook
 ┣ 📄 README.md               ← This file
 ┗ 📄 Mini_Project_...pdf     ← Project requirements
```

---

## 🚀 How to Run

1. Open the notebook in **Google Colab**
2. Run the dataset loading cell (requires Kaggle credentials)
3. Run EDA cells in order
4. Run preprocessing and model training cells
5. View evaluation results

Or click the badge at the top of the notebook:  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nawaf-Alorabi/-Classification_Mini_Project/blob/main/test.ipynb)

---

## 👥 Team Roles

| Role | Responsibility |
|---|---|
| Algorithm Explanation | How Random Forest works, advantages, limitations |
| **EDA & Visualization** | All charts, plots, and data insights ← *your part* |
| Data Preprocessing | Cleaning, encoding, scaling |
| Model Implementation | Sklearn + PyTorch custom model |
| Model Evaluation | Metrics, confusion matrix, interpretation |

---
