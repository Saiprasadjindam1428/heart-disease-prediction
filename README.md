# 🫀 Heart Disease Prediction using Logistic Regression

A machine learning project that predicts cardiovascular disease (CVD) risk using clinical patient data. The project covers end-to-end data analysis — from exploratory data analysis (EDA) to model training, evaluation, and hyperparameter tuning.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Key Findings from EDA](#key-findings-from-eda)
- [Model Performance](#model-performance)
- [Technologies Used](#technologies-used)
- [How to Run](#how-to-run)
- [Results Summary](#results-summary)

---

## Project Overview

This project uses **Logistic Regression** to classify whether a patient is at risk of heart disease based on 13 clinical features. The model is iteratively improved using **feature scaling** (StandardScaler) and **hyperparameter tuning** (GridSearchCV), boosting accuracy from **82% to 84%**.

---

## Dataset

The dataset contains **303 patient records** (302 after removing duplicates) with **13 input features** and 1 binary target variable.

| Feature | Description |
|---|---|
| `age` | Age of the patient (years) |
| `sex` | Sex (1 = Male, 0 = Female) |
| `cp` | Chest pain type (0–3) |
| `trestbps` | Resting blood pressure (mm Hg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl (1 = True) |
| `restecg` | Resting ECG results (0–2) |
| `thalach` | Maximum heart rate achieved (bpm) |
| `exang` | Exercise-induced angina (1 = Yes) |
| `oldpeak` | ST depression induced by exercise |
| `slope` | Slope of the peak exercise ST segment |
| `ca` | Number of major vessels colored by fluoroscopy (0–4) |
| `thal` | Thalassemia type (0–3) |
| `target` | **Heart disease risk** (1 = Diseased, 0 = Healthy) |

The target variable is **well-balanced** — approximately 54% diseased and 46% healthy.

---

## Project Workflow

```
1. Data Loading & Inspection
        ↓
2. Data Cleaning (duplicates, missing values)
        ↓
3. Exploratory Data Analysis (EDA)
        ↓
4. Feature & Target Separation
        ↓
5. Train/Test Split (80/20)
        ↓
6. Baseline Logistic Regression Model
        ↓
7. Feature Scaling (StandardScaler)
        ↓
8. Hyperparameter Tuning (GridSearchCV)
        ↓
9. Evaluation (Classification Report + Confusion Matrix)
```

---

## Key Findings from EDA

**Strongest predictors of heart disease:**

- **Chest Pain Type (`cp`)** — Non-anginal chest pain is strongly associated with higher CVD risk.
- **Max Heart Rate (`thalach`)** — Higher peak heart rate during exercise correlates with higher risk.
- **Thalassemia Type (`thal`)** — Type 2 is heavily associated with CVD (129 diseased vs. 36 healthy); Type 3 patients are mostly healthy.
- **Number of Major Vessels (`ca`)** — More clear vessels = lower risk.
- **Exercise-Induced Angina (`exang`)** — Patients *without* exercise-induced angina are paradoxically more often in the at-risk group.

**Surprising non-predictors:**

- **Cholesterol (`chol`)** and **Fasting Blood Sugar (`fbs`)** showed near-zero correlation with the target in this dataset — contrary to common assumptions.
- **Resting Blood Pressure (`trestbps`)** — Median values were almost identical across both groups (~130 mm Hg).

**Demographics:**

- Dataset is skewed towards males (68.2% male, 31.8% female).
- Heart disease risk peaks for patients in their **early-to-mid 50s**; past age 60, the trend reverses in this dataset.

---

## Model Performance

### Baseline Logistic Regression

| Metric | Score |
|---|---|
| Overall Accuracy | **82%** |
| True Negatives (Healthy → Healthy) | 24 |
| True Positives (Diseased → Diseased) | 26 |
| False Positives (Healthy → Diseased) | 5 |
| False Negatives (Diseased → Healthy) | 6 |

### Tuned Logistic Regression (StandardScaler + GridSearchCV)

| Metric | Score |
|---|---|
| Overall Accuracy | **84%** |
| False Positives | Reduced from 5 → 4 |

The tuned model reduced false positives (healthy patients misclassified as diseased), which is important for reducing unnecessary medical interventions.

> ⚠️ **Note:** In a medical context, **False Negatives** (diseased patients predicted as healthy) are the most critical error. Minimising them should be the priority in any production deployment.

---

## Technologies Used

- **Python 3**
- **Pandas** — Data manipulation
- **Matplotlib & Seaborn** — Data visualisation
- **Scikit-learn** — Model building, scaling, and tuning
  - `LogisticRegression`
  - `StandardScaler`
  - `GridSearchCV`
  - `train_test_split`
  - `confusion_matrix`, `classification_report`
- **Google Colab** — Development environment

---

## How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/Saiprasadjindam1428/heart-disease-prediction.git
   cd heart-disease-prediction
   ```

2. **Install dependencies**
   ```bash
   pip install pandas matplotlib seaborn scikit-learn openpyxl
   ```

3. **Add the dataset**
   Place your `data.xlsx` file in the project directory and update the file path in the notebook:
   ```python
   df = pd.read_excel('data.xlsx')
   ```

4. **Run the notebook**
   Open `Heart_disease.ipynb` in Jupyter Notebook or Google Colab and run all cells.

---

## Results Summary

| Step | Accuracy |
|---|---|
| Baseline Logistic Regression | 82% |
| After Feature Scaling + GridSearchCV Tuning | **84%** |

The improvement highlights the importance of feature scaling for distance/gradient-based algorithms like Logistic Regression, and how systematic hyperparameter search can yield measurable gains even on a small dataset.

---

## 📌 Future Improvements

- Test additional classifiers (Random Forest, XGBoost, SVM)
- Address class imbalance more explicitly (SMOTE or class weighting)
- Build a simple prediction interface using Streamlit
- Prioritise recall for the "Diseased" class to reduce dangerous False Negatives

---
