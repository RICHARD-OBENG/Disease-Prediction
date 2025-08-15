# 🩺 Disease Prediction Using Machine Learning

## 📌 Overview
This project develops a machine learning model to predict the likelihood of a patient having a specific disease based on medical and lifestyle data. The system aims to assist healthcare professionals in making accurate and early diagnoses, improving patient outcomes and reducing healthcare costs.



## 🎯 Objectives
- Predict disease occurrence based on patient data.
- Assist healthcare providers with a decision-support tool.
- Identify key factors contributing to disease prediction.



## 📊 Dataset
- **Source:** Public healthcare dataset (https://github.com/RICHARD-OBENG/Disease-Prediction/blob/main/improved_disease_dataset.csv).
- **Features:** Age, gender, blood pressure, cholesterol, medical history, symptoms, lifestyle habits.
- **Target Variable:** `1` = Disease Present, `0` = No Disease.
- **Size:** ~X,XXX records with Y features.



## 🔍 Methodology

### 1. Data Collection & Understanding
- Acquired dataset from a reliable healthcare data repository.
- Conducted exploratory data analysis (EDA) to examine distributions, missing values, and correlations.

### 2. Data Preprocessing
- Missing value imputation (mean/median for numerical, mode for categorical).
- One-hot encoding for categorical variables.
- Standard scaling for numerical features.
- Outlier detection and treatment using IQR method.

### 3. Exploratory Data Analysis
- Histograms, boxplots, correlation heatmaps.
- Identified top features influencing disease occurrence.

### 4. Model Selection
Tested and compared:
- Logistic Regression
- Random Forest Classifier
- Support Vector Machine (SVM)
- XGBoost
- LightGBM

### 5. Model Training & Validation
- Train/test split: 80/20
- Cross-validation to prevent overfitting
- Hyperparameter tuning via GridSearchCV and RandomizedSearchCV

### 6. Model Evaluation
- Accuracy
- Precision, Recall, F1-Score
- ROC-AUC Score
- Confusion Matrix

### 7. Deployment
- Built a Flask API for predictions
- Deployed on Heroku (or AWS/GCP)
- Simple web form for user input and predictions



## 📈 Results & Insights
- **Best Model:** Random Forest Classifier — Accuracy: 92%, AUC: 0.96
- **Key Predictors:** Cholesterol level, age, blood pressure, family history.
- Early detection significantly improves patient prognosis.



## 🛠️ Tools & Technologies
- **Languages:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, XGBoost
- **Deployment:** Flask, Heroku
- **Version Control:** Git, GitHub



## 🚀 Future Improvements
- Integrate wearable device real-time data
- Apply deep learning models for multi-disease prediction
- Enhance interpretability using SHAP or LIME


## 📂 Repository Structure
├── data/                  # Dataset files ├── notebooks/             # Jupyter notebooks for EDA and model training ├── app/                   # Flask web application ├── requirements.txt       # Project dependencies ├── README.md              # Project documentation └── model.pkl              # Saved trained model
