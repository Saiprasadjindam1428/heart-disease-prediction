# 🫀 Heart Disease Prediction using Logistic Regression

A machine learning project that predicts cardiovascular disease (CVD) risk using clinical patient data. Recently refactored from a Jupyter Notebook into a **production-ready software architecture**, this project covers end-to-end data analysis, automated visualizations, hyperparameter tuning, and persistent model saving.

---

## 📋 Table of Contents

- [Project Overview](#Project-Overview)
- [Dataset](#Dataset)
- [Project Architecture & Workflow](#Project-Architecture-and-Workflow)
- [Key Findings & Feature Importance](#key-findings--feature-importance)
- [Model Performance](#model-performance)
- [Technologies Used](#technologies-used)
- [How to Run](#how-to-run)
- [Results Summary](#results-summary)
- [Future Improvements](#future-improvements)

---

## Project Overview

This project uses **Logistic Regression** to classify whether a patient is at risk of heart disease based on 13 clinical features. The model is built inside a fully modular Python pipeline controlled by a `config.yaml` file. It iteratively improves using **feature scaling** (StandardScaler) and **hyperparameter tuning** (GridSearchCV), achieving an accuracy of **85%**.

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

## Project Architecture and Workflow

The code has been upgraded from a linear notebook into a professional repository structure:

1. `config/config.yaml` (Master setup for hyperparameters and paths)
2. `src/data_loader.py` (Data ingestion and cleaning)
3. `src/visualizer.py` (Automated EDA generation -> outputs/)
4. `src/model.py` (GridSearchCV, scaling, training & joblib saving -> models/)
5. `src/evaluate.py` (Classification reports, ROC, feature weights -> outputs/)

---

## Key Findings & Feature Importance

By extracting the mathematical weights (coefficients) from the tuned Logistic Regression model, the pipeline identified the **Top 5 strongest predictors** of heart disease:

1. **Number of Major Vessels (`ca`)**: More clear vessels strongly indicate a *lower* risk.
2. **Sex (`sex`)**: Gender plays a significant mathematical role in the model's risk calculation.
3. **Chest Pain Type (`cp`)**: Non-anginal chest pain is strongly associated with higher CVD risk.
4. **Thalassemia (`thal`)**: Type 2 is heavily associated with disease; Type 3 patients are mostly healthy.
5. **ST Depression (`oldpeak`)**: A critical ECG finding indicating cardiovascular stress.

**Surprising non-predictors:**
**Cholesterol (`chol`)** and **Fasting Blood Sugar (`fbs`)** showed near-zero mathematical correlation with the target in this specific dataset — contrary to common assumptions.

---

## Model Performance

The model was evaluated on a 20% holdout test set (61 patients). 

### Tuned Logistic Regression (StandardScaler + GridSearchCV)

| Metric | Score |
|---|---|
| Overall Accuracy | **85%** |
| Precision (Diseased) | 87% |
| Recall (Diseased) | 84% |
| True Negatives (Healthy → Healthy) | 25 |
| True Positives (Diseased → Diseased) | 27 |
| False Positives (Healthy → Diseased) | 4 |
| False Negatives (Diseased → Healthy) | 5 |

The tuned model successfully reduced false positives. 

> ⚠️ **Note:** In a medical context, **False Negatives** (diseased patients predicted as healthy) are the most critical error. The model correctly caught 27 out of 32 diseased patients in the test set.

---

## Technologies Used

- **Python 3**
- **Pandas & OpenPyXL** — Data manipulation and ingestion
- **Matplotlib & Seaborn** — Automated data visualization
- **Scikit-learn** — Model building, scaling, pipelines, and tuning
- **PyYAML** — Configuration management
- **Joblib** — Model serialization and saving
- **Custom Logger** — Enterprise-grade terminal and file logging

---

## How to Run

1. **Clone the repository**
   git clone https://github.com/Saiprasadjindam1428/heart-disease-prediction.git
   cd heart-disease-prediction

2. **Install dependencies**
   pip install -r requirements.txt

3. **Add the dataset**
   Place your `data.xlsx` file inside the `data/` folder. *(Note: The data folder is ignored by git for privacy and size constraints).*

4. **Run the pipeline**
   python main.py
   *(The script will automatically generate your logs, charts in the `outputs/` folder, and save the final model in the `models/` folder.)*

---

## Results Summary

| Step | Accuracy |
|---|---|
| Baseline Logistic Regression (Notebook) | 82% |
| Modular Pipeline with GridSearchCV Tuning | **85%** |

The transition to a formal pipeline allows for rapid parameter testing, yielding measurable gains in accuracy and creating a deployable `.joblib` asset.

---

## 📌 Future Improvements

- Test additional classifiers (Random Forest, XGBoost) directly within `model.py`.
- Build a prediction interface using Streamlit that loads the saved `best_logistic_model.joblib` file.
- Adjust prediction thresholds to prioritize recall for the "Diseased" class, reducing dangerous False Negatives further.
