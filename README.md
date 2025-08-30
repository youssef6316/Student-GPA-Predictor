# 🎓 Student GPA Predictor

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red)](https://streamlit.io/)

## 📌 Project Overview

A **machine learning app** that predicts a student’s **GPA value** based on academic lifestyle and personal features.
The project also provides **insights & visualizations** to analyze factors affecting student performance, making it useful for **risk analysis** and **early intervention**.

---

## 🏫 Motivation

This project was developed during my training at the **National Telecommunication Institute (NTI)**.
The goal was to not only practice machine learning but also to address a real-world issue: **avoiding student performance drop** by identifying risk factors early and extracting meaningful insights.

---

## ✨ Features

* ✅ **Interactive Streamlit App** with two tabs:

  * **Prediction Tab** → Input student details and get GPA prediction.
  * **Analysis Tab** → Explore insights, visualizations, and model performance.
* ✅ Multiple ML models trained and compared.
* ✅ Preprocessing & feature engineering pipeline.
* ✅ Rich set of visualizations for **data understanding** and **model evaluation**.

---

## 🛠️ Tech Stack

* **Language:** Python
* **Libraries & Tools:**

  * Data Handling → `pandas`, `numpy`
  * Visualization → `matplotlib`, `seaborn`
  * Machine Learning → `scikit-learn` (Linear Regression, Decision Tree, Random Forest, SVR, KNN)
  * Model Persistence → `joblib`
  * UI → `streamlit`, `streamlit-lottie`
  * Notebook Experiments → Jupyter Notebook

---

## 📊 Dataset

* **Source:** [Kaggle](https://www.kaggle.com/) (synthetic dataset for research & educational purposes).
* **Size:** 5,000 records of high school students.
* **Features Include:**

  * **Demographics:** Age, Gender, Ethnicity, Parental Education.
  * **Study Habits:** Weekly Study Hours, Absences, Tutoring.
  * **Parental Involvement:** Parental Support.
  * **Activities:** Extracurricular, Sports, Music, Volunteering.
  * **Academic Performance:** Grade Class (A–D), GPA (Target).

---

## 🧑‍💻 Data Preprocessing & Feature Engineering

* ✅ Cleaned data → removed duplicates, outliers, missing values.
* ✅ Renamed columns for clarity.
* ✅ Rounded GPA values (2 decimals).
* ✅ Engineered 6 new features:
    `Study_Efficiency`, `Absence_Rate`, `Absence_Impact`, `Study_Impact`, `Is_Child`, `Absence_Rate_Category`, `Total_Activities`.
* ✅ Normalized GPA target (square transform).
* ✅ Feature encoding using **ColumnTransformer** (binary & multi-class categories).
* ✅ Feature selection → Based on Random Forest importance.

---

## 🤖 Models & Results

| Model             | Train R² | Test R² | Prediction Time (ms) |
| ----------------- | -------- | ------- | -------------------- |
| Linear Regression | 0.718    | 0.715   | 0.3413               |
| SVR               | 0.989    | 0.983   | 0.1576               |
| KNN               | 1.000    | 0.840   | 1.0698               |
| Random Forest     | 0.999    | 0.995   | 5.8188               |
| Decision Tree     | 0.997    | 0.985   | 0.5271               |

* 📌 **Best performer:** **Random Forest Regressor** (highest accuracy, robust results).
* 📌 Trade-off → SVR was faster, RF was more accurate.

---

## 🚀 How to Run

### 🔹 Run the Jupyter Notebook

1. Open `Regression.ipynb`.
2. Run cells sequentially to reproduce preprocessing, modeling, and evaluation.

### 🔹 Run the Streamlit App

```bash
# Go to project root
cd Student_performance_project  

# Run the app
streamlit run app/app.py
```

---

## 📂 Project Structure

```
Student_performance_project/
│
├── app/
│   └── app.py                # Streamlit app (prediction + analysis UI)
│
├── Data/
│   ├── students_data.csv      # Raw dataset
│   └── student_data_clean.csv # Cleaned dataset
│
├── Models/
│   ├── best_reg_model.joblib  # Saved best model
│   └── reg_preprocessor.joblib# Preprocessing pipeline
│
├── Regression.ipynb           # Main notebook with EDA & modeling
└── README.md                  # Project description (this file)
```

---

## 🌱 Future Improvements

*  Deploy the Streamlit app online (Streamlit Cloud / Heroku / Render).
*  Add a **classification module** for predicting **Grade Class (A–F)** alongside GPA regression.
*  Enhance interpretability with **SHAP/Explainable AI** for feature importance.
*  Expand dataset with more **real-world student data**.
*  Add a **real-time API** (FastAPI/Flask) for predictions.
*  Optimize inference speed with **model compression** (e.g., pruning, quantization).

---

## 🙏 Acknowledgments

* **Dataset:** Kaggle synthetic dataset (educational purpose).
* **Training Support:** National Telecommunication Institute (NTI).
* **Streamlit:** A faster way to build and share data apps

---
