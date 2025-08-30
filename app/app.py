import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import datetime
import time
from streamlit_lottie import st_lottie
import requests

import os

def save_user_data(user_data, filename="user_logs.csv"):
    if os.path.exists(filename):
        df = pd.read_csv(filename)
    else:
        df = pd.DataFrame()

    df = pd.concat([df, pd.DataFrame([user_data])], ignore_index=True)
    df.to_csv(filename, index=False)


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
model = joblib.load(os.path.join(os.path.dirname(__file__), "..", "Models", "best_reg_model.joblib"))
# Sidebar legend with navigation
st.sidebar.title("📌 Menu")
page = st.sidebar.radio("Choose a Section", ["🔮 Predictions", "📊 Plots & Insights"])

if page == "🔮 Predictions":

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

            user_data = {
                "timestamp": datetime.datetime.now(),
                "username": name,
                "GPA": gpa,
                "age": age,
                "study_hours": study_time,
                "absences": absences
            }

            # Save to CSV (replace with DB in production)
            df = pd.DataFrame([user_data])
            df.to_csv("user_data.csv", mode="a", header=not pd.io.common.file_exists("user_data.csv"), index=False)

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

    df = pd.read_csv(r"Data/student_data_clean.csv")

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

        # Detect categorical vs numerical
        if pd.api.types.is_numeric_dtype(df[selected_feature]):
            feature_type = "numeric"
        else:
            feature_type = "categorical"

        # --- Hue selection (only for categorical features, and hue itself must be categorical) ---
        if feature_type == "categorical":
            categorical_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
            categorical_cols = [c for c in categorical_cols if c != selected_feature]  # exclude the main feature itself

            if categorical_cols:  # only show if there are categorical options
                selected_hue = st.selectbox(
                    "🎨 Choose a Hue (optional):",
                    [None] + categorical_cols,
                    key="feat2"
                )

        # --- Plot ---
        fig, ax = plt.subplots(figsize=(8, 4))

        if feature_type == "categorical":
            sns.countplot(
                x=selected_feature,
                data=df,
                hue=selected_hue if selected_hue else None,
                order=sorted(df[selected_feature].astype(str).unique()),
                ax=ax,
                palette="winter"  # better than one static color
            )
            ax.set_title(f"Distribution by {selected_feature}")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=30)

        else:  # numeric
            sns.histplot(
                df[selected_feature],
                bins=20,
                kde=True,
                ax=ax,
                palette="winter"
            )
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
                "📈 Females have a higher share in high grades than Males."
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
            "Linear Regression": {"Best train R²": 0.718,
                                  "Best test R²": 0.716,
                                  "Prediction time (ms)": 0.3413},
            "SVR": {"Best train R²": 0.897,
                    "Best test R²": 0.894,
                    "Prediction time (ms)": 1.1576},
            "KNN": {"Best train R²": 1.0,
                    "Best test R²": 0.84,
                    "Prediction time (ms)": 1.0698},
            "Random Forest": {"Best train R²": 0.999,
                              "Best test R²": 0.995,
                              "Prediction time (ms)": 5.8188},
            "Decision Tree": {"Best train R²": 0.997,
                              "Best test R²": 0.986,
                              "Prediction time (ms)": 0.5271}
        }

        # --- Accuracy DataFrame ---
        accuracy_scores = {model: {"Best train R²": vals["Best train R²"],
                                   "Best test R²": vals["Best test R²"]}
                           for model, vals in model_scores.items()}

        score_df = pd.DataFrame(accuracy_scores).T
        st.dataframe(score_df)

        # Melt for seaborn long-format plotting
        score_melted = score_df.reset_index().melt(id_vars="index",
                                                   value_vars=["Best train R²", "Best test R²"],
                                                   var_name="Metric", value_name="Score")
        score_melted.rename(columns={"index": "Model"}, inplace=True)

        fig1, ax1 = plt.subplots(figsize=(8, 8))
        sns.barplot(data=score_melted, x="Model", y="Score", hue="Metric", palette="winter", ax=ax1)
        ax1.set_title("Model Accuracy (Train vs Test)")
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=30)
        st.pyplot(fig1)

        # --- Timing DataFrame ---
        timing_scores = {
            model: {"Prediction time (ms)": vals["Prediction time (ms)"]}
            for model, vals in model_scores.items()
        }

        time_df = pd.DataFrame(timing_scores).T
        st.dataframe(time_df)

        time_melted = time_df.reset_index().melt(id_vars="index",
                                                 value_vars=["Prediction time (ms)"],
                                                 var_name="Metric", value_name="Time (ms)")
        time_melted.rename(columns={"index": "Model"}, inplace=True)

        fig2, ax2 = plt.subplots(figsize=(8, 8))
        sns.barplot(data=time_melted, x="Model", y="Time (ms)", palette="winter", ax=ax2)
        ax2.set_title("Prediction Time (ms)")
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=30)
        st.pyplot(fig2)

# ========================
# Global Footer
# ========================

footer = """
<style>
footer {
    visibility: hidden;
}
div[data-testid="stSidebar"] {
    z-index: 999;
}
div.block-container {
    padding-bottom: 0px; 
}
div[data-testid="stFooter"] {
    display: none;
}
.full-width-footer {
    position: relative;
    left: 0;
    width: 100%;
    margin: 0;
    padding: 20px 0;
    background-color: #;
    color: white;
    text-align: center;
    font-size: 16px;
}
.full-width-footer p {
    margin: 6px 0;
}
</style>

<div class="full-width-footer">
    <p>🎓 <b>Student Performance Predictor</b> | Exploring academic success through data</p>
    <p>💡 Built with ❤️ using <b>Streamlit</b> & <b>Machine Learning</b></p>
    <p>👨‍💻 Developed by <b>Youssef Yacoub</b></p>
</div>
"""

import streamlit as st
st.markdown(footer, unsafe_allow_html=True)
