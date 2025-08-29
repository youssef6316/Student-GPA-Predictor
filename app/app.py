import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time
from streamlit_lottie import st_lottie
import requests


# Helper to load Lottie animations
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


# Load animations
success_anim = load_lottieurl("https://assets4.lottiefiles.com/packages/lf20_touohxv0.json")
chart_anim = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_qp1q7mct.json")

# Load model
model = joblib.load("best_reg_model.joblib")

# Sidebar legend with navigation
st.sidebar.title("📌 Menu")
page = st.sidebar.radio("Choose a Section", ["🔮 Predictions", "📊 Plots & Insights"])

st.title("🎓 Student Performance Predictor")

# 1️⃣ Input fields
name = st.text_input("Name", placeholder="Enter your full name")
age = st.text_input("Age", placeholder="Enter age in years")
study_time = st.text_input("Weekly Study Time (hours)",
                           placeholder="Daily studying hours * count of studying days")
absences = st.text_input("Absences",
                         placeholder="Number of absences per Month (0 - 30) days")
gpa = st.text_input("GPA", placeholder="Final GPA out of 4.0, e.g. 3.75")

# Initialize session state
if "valid" not in st.session_state:
    st.session_state.valid = False
if "inputs" not in st.session_state:
    st.session_state.inputs = {}

# 1️⃣ Submit button for validation
if st.button("Submit"):
    valid = True
    inputs = {}

    # --- Name ---
    if not name.strip():
        st.error("❌ Name cannot be empty")
        valid = False

    # --- Age ---
    try:
        inputs["age"] = int(age)
        if inputs["age"] < 15 or inputs["age"] > 100:
            st.error("❌ Age must be between 15 and 100")
            valid = False
    except:
        st.error("❌ Age must be an Integer Number")
        valid = False

    # --- Study time ---
    try:
        inputs["study_time"] = int(study_time)
        if inputs["study_time"] <= 0:
            st.error("❌ Weekly Study Time must be higher than 0")
            valid = False
    except:
        st.error("❌ Weekly Study Time must be an Integer Number")
        valid = False

    # --- Absences ---
    try:
        inputs["absences"] = int(absences)
        if inputs["absences"] < 0 or inputs["absences"] > 30:
            st.error("❌ Absences must be between 0 and 30")
            valid = False
    except:
        st.error("❌ Absences must be an Integer Number")
        valid = False

    # --- GPA ---
    try:
        inputs["gpa"] = float(gpa)
        if inputs["gpa"] < 0 or inputs["gpa"] > 4:
            st.error("❌ GPA must be between 0.0 and 4.0")
            valid = False
    except:
        st.error("❌ GPA must be a Number (e.g., 3.5)")
        valid = False

    # ✅ Save results in session state
    st.session_state.valid = valid
    if valid:
        st.session_state.inputs = inputs
        st.success("✅ All inputs are valid, ready for prediction!")

# 2️⃣ Prediction with animation & decoded class
if st.button("Predict Performance 🚀"):
    if not st.session_state.valid:
        st.error("⚠️ Please validate inputs first using Submit")
    else:
        # Fetch validated inputs
        inputs = st.session_state.inputs
        study_time = inputs["study_time"]
        absences = inputs["absences"]
        gpa = inputs["gpa"]

        # Feature engineering
        absence_rate = absences / 30
        absence_impact = gpa / absences if absences else gpa / 0.01
        study_impact = study_time * gpa if study_time else study_time * 0.01

        input_df = pd.DataFrame([{
            "remainder__Weekly_Study_Time": study_time,
            "remainder__Absences": absences,
            "remainder__Absence_Rate": absence_rate,
            "remainder__Absence_Impact": absence_impact,
            "remainder__Study_Impact": study_impact
        }])

        with st.spinner("Analyzing student performance..."):
            time.sleep(2)  # Fake loading
            pred = model.predict(input_df)
            pred = np.sqrt(pred.astype(float))
            pred = np.round(pred, 3)

        st.markdown(
            f"<h3 style='color:green;'>📊 Predicted Student Performance: {pred}</h3>",
            unsafe_allow_html=True
        )


elif page == "📊 Plots & Insights":
    st.title("📊 Interactive Data & Model Insights")

    # Winter-dark theme colors
    sns.set_theme(style="darkgrid", palette="winter")

    st_lottie(chart_anim, height=180)
    st.markdown("### 🔍 Explore the Dataset & Model Performance")

    df = pd.read_csv("student_data_clean.csv")

    tabs = st.tabs(["📈 Data Overview", "🤖 Model Comparison"])

    # =====================
    # TAB 1: DATA OVERVIEW
    # =====================
    with tabs[0]:
        st.subheader("📊 Data Insights & Visualizations")
        st.markdown("---")

        # --- Feature selection ---
        all_features = [str(c) for c in df.columns]  # ensure all columns are strings
        selected_feature = st.selectbox(
            "🔎 Choose a feature to explore:",
            all_features,
            key="feat1"
        )

        # --- Plot ---
        fig, ax = plt.subplots(figsize=(8, 4))

        if selected_feature not in ['Age', 'Weekly_Study_Time', 'Absences', 'GPA', 'Study_Efficiency', 'Absence_Rate',
                                    'Absence_Impact', 'Study_Impact']:
            # Categorical → Count plot
            sns.countplot(
                x=selected_feature,
                data=df,
                order=sorted(df[selected_feature].unique()),
                ax=ax,
                color="#FFa890"
            )
            ax.set_title(f"Distribution by {selected_feature}")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
        else:
            # Numerical → Histogram + KDE
            sns.histplot(df[selected_feature], bins=20, kde=True, ax=ax, color="#FFa890")
            ax.set_title(f"Distribution of {selected_feature}")

        st.pyplot(fig)

        # --- Insights ---
        # You can customize insights per feature if you want
        insights = {
            "age": [
                "👨‍👩‍👧 Average age of each gender is ~16.",
                "🎭 16-year-olds are the most engaged in extracurricular activities."
            ],
            "gender": [
                "📈 Males have a higher share in high grades than females."
            ],
            "tutoring": [
                "📚 Non-tutored students are more likely to score higher."
            ],
            "parent_support": [
                "🤝 Greater parental support increases the probability of higher grades."
            ],
            "ethnicity": [
                "🌏 Asians tend to be the weakest scorers among ethnicities."
            ],
            "grades_inheritance": [
                "🧬 High grades are not inherited — children of parents with top academic records often score lower."
            ]
        }

        for category, insights_list in insights.items():
            st.markdown(f"### {category}")
            for insight in insights_list:
                st.markdown(f"""
                        <div style="
                            background-color:#f8f9fa;
                            color:#212529;
                            padding:10px;
                            border-radius:10px;
                            margin:5px 0;
                            box-shadow: 0px 2px 6px rgba(0,0,0,0.1);
                            font-size:15px;
                            text-align:center;">
                            {insight}
                        </div>
                    """, unsafe_allow_html=True)

    # ==========================
    # TAB 2: MODEL COMPARISON
    # ==========================
    with tabs[1]:
        st.markdown("#### 📊 Model Performance Comparison")

        # Load performance metrics from CSV / dictionary
        model_scores = {
            "Linear Regression": {"Best train R²": 0.7181248240935445,
                                  "Best test R²": 0.7159529834070468,
                                  "Prediction time (ms)": 0.3413},
            "SVR": {"Train Accuracy": 0.989,
                    "Test Accuracy": 0.983,
                    "Train Loss": 0.037,
                    "Test Loss": 0.044,
                    "Prediction time (ms)": 0.1576},
            "KNN": {"Best train R²": 1.0,
                    "Best test R²": 0.8400238399557498,
                    "Prediction time (ms)": 1.0698},
            "Random Forest": {"Best train R²": 0.9992550689769562,
                              "Best test R²": 0.9950128458208851,
                              "Prediction time (ms)": 5.8188},
            "Decision Tree": {"Best train R²": 0.9973320525368343,
                              "Best test R²": 0.9855900670692451,
                              "Prediction time (ms)": 0.5271}
        }

        # --- Accuracy DataFrame ---
        accuracy_scores = {model: {"Train Accuracy": vals["Train Accuracy"],
                                   "Test Accuracy": vals["Test Accuracy"]}
                           for model, vals in model_scores.items()}

        score_df = pd.DataFrame(accuracy_scores).T
        st.dataframe(score_df)

        fig1, ax1 = plt.subplots()
        score_df.plot(kind="bar", ax=ax1, color="#FFa890")
        ax1.set_title("Model Accuracy")
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=30)
        st.pyplot(fig1)

        # --- Timing DataFrame ---
        timing_scores = {model: {"Prediction time (ms)": vals["Prediction time (ms)"]}
                         for model, vals in model_scores.items()}

        time_df = pd.DataFrame(timing_scores).T
        st.dataframe(time_df)

        fig, ax = plt.subplots()
        time_df.plot(kind="bar", ax=ax, color="#FFa890")
        ax.set_title("Prediction Time (ms)")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
        st.pyplot(fig)
