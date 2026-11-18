
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from datetime import datetime
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)

# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "dataset" / "cleaned_hr_attrition.csv"
MODEL_DIR = BASE_DIR / "models"


# ==========================================
# 2. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="HR Employee Attrition Prediction",
    page_icon="👨‍💼",
    layout="wide"
)


# ==========================================
# 3. LOAD MODEL AND DATA
# ==========================================

@st.cache_resource
def load_assets():

    df = pd.read_csv(DATA_PATH)

    model_path = MODEL_DIR / "optimized_model.pkl"

    if not model_path.exists():
        model_path = MODEL_DIR / "best_model.pkl"

    model = joblib.load(model_path)

    preprocessor = joblib.load(
        MODEL_DIR / "preprocessor.pkl"
    )

    data = joblib.load(
        MODEL_DIR / "encoded_data.pkl"
    )

    return df, model, preprocessor, data


try:

    df, model, preprocessor, data = load_assets()

except Exception as e:

    st.error(f"Project Loading Error: {e}")
    st.stop()


# ==========================================
# 4. SESSION STATE
# ==========================================

if "prediction_history" not in st.session_state:

    st.session_state.prediction_history = []


# ==========================================
# 5. MAIN HEADER
# ==========================================

st.title("👨‍💼 HR Employee Attrition Prediction")

st.write(
    "Machine Learning Based Employee Attrition "
    "Analysis and Prediction System"
)

st.divider()


# ==========================================
# 6. SIDEBAR NAVIGATION
# ==========================================

st.sidebar.title("HR Management System")

page = st.sidebar.radio(
    "Select Page",
    [
        "Dashboard",
        "Predict Attrition",
        "Model Performance",
        "Prediction History"
    ]
)

st.sidebar.divider()

st.sidebar.info(
    "HR Employee Attrition Prediction "
    "using Machine Learning."
)


# ==========================================
# 7. DASHBOARD PAGE
# ==========================================

if page == "Dashboard":

    st.header("📊 HR Analytics Dashboard")

    total_employees = len(df)

    employees_left = int(
        (df["Attrition"] == 1).sum()
    )

    employees_stayed = int(
        (df["Attrition"] == 0).sum()
    )

    attrition_rate = (
        employees_left / total_employees
    ) * 100

    # Dashboard metrics

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Employees",
        total_employees
    )

    col2.metric(
        "Employees Left",
        employees_left
    )

    col3.metric(
        "Employees Stayed",
        employees_stayed
    )

    col4.metric(
        "Attrition Rate",
        f"{attrition_rate:.2f}%"
    )

    st.divider()

    # Attrition distribution

    st.subheader("Employee Attrition Distribution")

    chart_data = pd.DataFrame({
        "Status": ["Stayed", "Left"],
        "Employees": [
            employees_stayed,
            employees_left
        ]
    })

    col1, col2 = st.columns(2)

    with col1:

        st.bar_chart(
            chart_data.set_index("Status")
        )

    with col2:

        fig, ax = plt.subplots(figsize=(6, 4))

        ax.pie(
            chart_data["Employees"],
            labels=chart_data["Status"],
            autopct="%1.1f%%",
            startangle=90
        )

        ax.set_title("Attrition Percentage")

        st.pyplot(fig)

        plt.close(fig)

    st.divider()

    # Department-wise analysis

    st.subheader("Department-wise Attrition")

    department_data = (
        df.groupby("Department")["Attrition"].mean() * 100
    )

    st.bar_chart(department_data)

    # Overtime analysis

    st.subheader("Overtime vs Attrition")

    overtime_data = pd.crosstab(
        df["OverTime"],
        df["Attrition"]
    )

    overtime_data.columns = [
        "Stayed",
        "Left"
    ]

    st.bar_chart(overtime_data)

    st.success("HR Analytics Dashboard Loaded Successfully!")


# ==========================================
# 8. PREDICTION PAGE
# ==========================================

elif page == "Predict Attrition":

    st.header("🔍 Employee Attrition Prediction")

    st.write(
        "Enter employee details to predict "
        "employee attrition."
    )

    feature_columns = df.drop(
        columns=["Attrition"]
    ).columns.tolist()

    categorical_columns = df[
        feature_columns
    ].select_dtypes(
        include=["object"]
    ).columns.tolist()

    input_data = {}

    st.subheader("Employee Information")

    with st.form("employee_form"):

        col1, col2, col3 = st.columns(3)

        for index, feature in enumerate(feature_columns):

            if index % 3 == 0:
                current_col = col1

            elif index % 3 == 1:
                current_col = col2

            else:
                current_col = col3

            with current_col:

                if feature in categorical_columns:

                    options = sorted(
                        df[feature].dropna().unique().tolist()
                    )

                    input_data[feature] = st.selectbox(
                        feature,
                        options
                    )

                else:

                    minimum = int(df[feature].min())

                    maximum = int(df[feature].max())

                    default = int(df[feature].median())

                    input_data[feature] = st.number_input(
                        feature,
                        min_value=minimum,
                        max_value=maximum,
                        value=default,
                        step=1
                    )

        submitted = st.form_submit_button(
            "🔍 Predict Attrition",
            use_container_width=True
        )

    if submitted:

        input_df = pd.DataFrame(
            [input_data],
            columns=feature_columns
        )

        try:

            transformed_data = preprocessor.transform(
                input_df
            )

            if isinstance(
                model,
                GradientBoostingClassifier
            ):

                if hasattr(transformed_data, "toarray"):

                    transformed_data = transformed_data.toarray()

            prediction = model.predict(
                transformed_data
            )[0]

            probability = model.predict_proba(
                transformed_data
            )[0][1]

            # Save prediction record

            record = input_df.iloc[0].to_dict()

            record["Prediction"] = (
                "Leave" if prediction == 1 else "Stay"
            )

            record["Attrition_Probability"] = round(
                probability * 100, 2
            )

            record["Prediction_Date"] = (
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )

            st.session_state.prediction_history.append(
                record
            )

            # Display result

            st.divider()

            st.subheader("📊 Prediction Result")

            col1, col2 = st.columns(2)

            col1.metric(
                "Attrition Probability",
                f"{probability * 100:.2f}%"
            )

            col2.metric(
                "Prediction",
                "Leave" if prediction == 1 else "Stay"
            )

            if prediction == 1:

                st.error(
                    "⚠️ Employee is predicted to leave."
                )

            else:

                st.success(
                    "✅ Employee is predicted to stay."
                )

            # Risk classification

            if probability < 0.30:

                st.success("Risk Level: LOW")

            elif probability < 0.60:

                st.warning("Risk Level: MEDIUM")

            else:

                st.error("Risk Level: HIGH")

            st.progress(float(probability))

            st.caption(
                "Prediction is a statistical estimate, "
                "not a certainty or employment decision."
            )

        except Exception as e:

            st.error(f"Prediction Error: {e}")


# ==========================================
# 9. MODEL PERFORMANCE PAGE
# ==========================================

elif page == "Model Performance":

    st.header("🤖 Model Performance Analysis")

    X_test = data["X_test"]

    y_test = data["y_test"]

    if isinstance(model, GradientBoostingClassifier):

        if hasattr(X_test, "toarray"):

            X_test = X_test.toarray()

    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(
        y_test, y_pred
    )

    precision = precision_score(
        y_test, y_pred, zero_division=0
    )

    recall = recall_score(
        y_test, y_pred, zero_division=0
    )

    f1 = f1_score(
        y_test, y_pred, zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test, y_prob
    )

    st.subheader("Evaluation Metrics")

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Accuracy", f"{accuracy*100:.2f}%")

    col2.metric("Precision", f"{precision*100:.2f}%")

    col3.metric("Recall", f"{recall*100:.2f}%")

    col4.metric("F1 Score", f"{f1*100:.2f}%")

    col5.metric("ROC-AUC", f"{roc_auc:.3f}")

    st.divider()

    # Confusion Matrix

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Stayed", "Left"],
        yticklabels=["Stayed", "Left"],
        ax=ax
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)

    plt.close(fig)

    st.divider()

    # Model Comparison

    st.subheader("Model Comparison")

    comparison_path = MODEL_DIR / "model_comparison.csv"

    if comparison_path.exists():

        results = pd.read_csv(comparison_path)

        st.dataframe(
            results.sort_values(
                by="F1 Score",
                ascending=False
            ),
            use_container_width=True,
            hide_index=True
        )

    st.success("Model Evaluation Completed!")


# ==========================================
# 10. PREDICTION HISTORY PAGE
# ==========================================

elif page == "Prediction History":

    st.header("📋 Employee Prediction History")

    history = st.session_state.prediction_history

    if history:

        history_df = pd.DataFrame(history)

        col1, col2 = st.columns(2)

        col1.metric(
            "Total Predictions",
            len(history_df)
        )

        col2.metric(
            "Predicted Attrition",
            int((history_df["Prediction"] == "Leave").sum())
        )

        st.divider()

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        # Download CSV

        csv = history_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="📥 Download HR Report",
            data=csv,
            file_name="HR_Attrition_Report.csv",
            mime="text/csv",
            use_container_width=True
        )

        if st.button("Clear Prediction History"):

            st.session_state.prediction_history = []

            st.rerun()

    else:

        st.info(
            "No prediction history available. "
            "Please make an employee prediction first."
        )


# ==========================================
# 11. FOOTER
# ==========================================

st.divider()

st.caption(
    "HR Employee Attrition Prediction | "
    "Machine Learning Project"
)

st.caption(
    "Developed using Python, Scikit-learn, "
    "Pandas and Streamlit."
)
