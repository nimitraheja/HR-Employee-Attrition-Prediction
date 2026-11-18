
import streamlit as st
import pandas as pd
import joblib

from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset" / "cleaned_hr_attrition.csv"
MODEL_DIR = BASE_DIR / "models"

# Page configuration
st.set_page_config(
    page_title="HR Attrition Prediction",
    page_icon="👨‍💼",
    layout="wide"
)

# Load files
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

    return df, model, preprocessor


try:
    df, model, preprocessor = load_assets()
except Exception as e:
    st.error(f"Could not load project files: {e}")
    st.stop()


# Header
st.title("👨‍💼 HR Employee Attrition Prediction")
st.write(
    "Machine Learning based Employee Attrition Analysis System"
)

st.divider()

# Dashboard metrics
col1, col2, col3 = st.columns(3)

col1.metric("Total Employees", len(df))

col2.metric(
    "Employees Left",
    int((df["Attrition"] == 1).sum())
)

col3.metric(
    "Attrition Rate",
    f"{df['Attrition'].mean() * 100:.2f}%"
)

st.divider()

# Input form
st.header("📝 Employee Information")

feature_columns = df.drop(
    columns=["Attrition"]
).columns.tolist()

categorical_columns = df[
    feature_columns
].select_dtypes(include=["object"]).columns.tolist()

numeric_columns = [
    col for col in feature_columns
    if col not in categorical_columns
]

input_data = {}

with st.form("employee_form"):

    st.subheader("Enter Employee Details")

    col1, col2 = st.columns(2)

    for index, feature in enumerate(feature_columns):

        current_col = col1 if index % 2 == 0 else col2

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

                min_value = int(df[feature].min())
                max_value = int(df[feature].max())
                default_value = int(df[feature].median())

                input_data[feature] = st.number_input(
                    feature,
                    min_value=min_value,
                    max_value=max_value,
                    value=default_value,
                    step=1
                )

    submitted = st.form_submit_button(
        "🔍 Predict Attrition",
        use_container_width=True
    )


# Prediction
if submitted:

    input_df = pd.DataFrame(
        [input_data],
        columns=feature_columns
    )

    try:
        transformed_data = preprocessor.transform(input_df)

        if isinstance(model, GradientBoostingClassifier):
            if hasattr(transformed_data, "toarray"):
                transformed_data = transformed_data.toarray()

        prediction = model.predict(transformed_data)[0]

        probability = model.predict_proba(
            transformed_data
        )[0][1]

        st.divider()
        st.header("📊 Prediction Results")

        result_col1, result_col2 = st.columns(2)

        with result_col1:
            st.metric(
                "Attrition Probability",
                f"{probability * 100:.2f}%"
            )

        with result_col2:
            if prediction == 1:
                st.error("Employee predicted to leave")
            else:
                st.success("Employee predicted to stay")

        # Indicative risk bands
        if probability < 0.30:
            st.success("Risk Level: Low")
        elif probability < 0.60:
            st.warning("Risk Level: Medium")
        else:
            st.error("Risk Level: High")

        st.progress(float(probability))

        st.caption(
            "Risk bands are indicative thresholds, not validated HR policies. "
            "Predictions are estimates and should not determine employment decisions."
        )

    except Exception as e:
        st.error(f"Prediction error: {e}")


st.divider()

st.caption(
    "HR Employee Attrition Prediction | Machine Learning Project"
)
