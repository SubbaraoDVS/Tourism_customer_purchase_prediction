
import streamlit as st
import pandas as pd
import joblib
import os
import shutil
from pathlib import Path

# Define the base path for accessing files within the Colab environment on Google Drive
BASE_PROJECT_PATH_DRIVE = Path("/content/drive/My Drive/Colab_Notebooks/tourism_package_prediction")

# Define temporary local directory for Streamlit to access files
TEMP_STREAMLIT_DIR = Path("/content/streamlit_tmp")
TEMP_STREAMLIT_DIR.mkdir(parents=True, exist_ok=True)

# Define original paths in Google Drive
MODEL_PATH_DRIVE = BASE_PROJECT_PATH_DRIVE / "src" / "tourism_purchase_model.joblib"
TRAIN_DATA_PATH_DRIVE = BASE_PROJECT_PATH_DRIVE / "artifacts" / "train.csv"

# Define new local paths for the Streamlit app
MODEL_PATH_LOCAL = TEMP_STREAMLIT_DIR / "tourism_purchase_model.joblib"
TRAIN_DATA_PATH_LOCAL = TEMP_STREAMLIT_DIR / "train.csv"

# Copy files from Google Drive to local temporary directory
try:
    if not MODEL_PATH_LOCAL.exists():
        shutil.copy(MODEL_PATH_DRIVE, MODEL_PATH_LOCAL)
    st.success(f"Model copied to: {MODEL_PATH_LOCAL}")
except FileNotFoundError:
    st.error(f"Original model file not found in Drive at: {MODEL_PATH_DRIVE}. Please ensure the model is trained and saved.")
    MODEL_PATH_LOCAL = None # Indicate failure to copy

try:
    if not TRAIN_DATA_PATH_LOCAL.exists():
        shutil.copy(TRAIN_DATA_PATH_DRIVE, TRAIN_DATA_PATH_LOCAL)
    st.success(f"Train data copied to: {TRAIN_DATA_PATH_LOCAL}")
except FileNotFoundError:
    st.error(f"Original training data file not found in Drive at: {TRAIN_DATA_PATH_DRIVE}. Cannot infer feature types.")
    TRAIN_DATA_PATH_LOCAL = None # Indicate failure to copy

@st.cache_resource
def load_model():
    if MODEL_PATH_LOCAL and MODEL_PATH_LOCAL.exists():
        try:
            return joblib.load(MODEL_PATH_LOCAL)
        except Exception as e:
            st.error(f"Error loading model from local path: {e}")
            return None
    else:
        st.error("Model not available locally.")
        return None

@st.cache_data
def load_train_data():
    if TRAIN_DATA_PATH_LOCAL and TRAIN_DATA_PATH_LOCAL.exists():
        try:
            return pd.read_csv(TRAIN_DATA_PATH_LOCAL)
        except Exception as e:
            st.error(f"Error loading training data from local path: {e}")
            return pd.DataFrame()
    else:
        st.error("Training data not available locally. Cannot infer feature types.")
        return pd.DataFrame()

model = load_model()
train_df = load_train_data()

st.set_page_config(
    page_title="Tourism Purchase Prediction",
    page_icon="✈️",
    layout="wide"
)

st.title(
    "✈️ Tourism Package Purchase Prediction"
)

st.write(
    """
    Enter customer information to predict
    whether the customer is likely to purchase
    a tourism package.
    """
)

# --- Dynamic Input Form ---

if model is not None and not train_df.empty:
    TARGET_COLUMN = "ProdTaken"

    # Features used for training the model (excluding the target column)
    feature_columns = [
        column
        for column in train_df.columns
        if column != TARGET_COLUMN
    ]

    input_data = {}
    st.subheader("Customer Details")

    # Organize inputs into two columns
    col1, col2 = st.columns(2)

    for i, column in enumerate(feature_columns):
        if i % 2 == 0: # Place in first column
            with col1:
                if train_df[column].dtype == "object":
                    options = (
                        train_df[column]
                        .dropna()
                        .unique()
                        .tolist()
                    )
                    input_data[column] = st.selectbox(column, options)
                else:
                    median_value = float(
                        train_df[column]
                        .median()
                    )
                    input_data[column] = st.number_input(column, value=median_value)
        else: # Place in second column
            with col2:
                if train_df[column].dtype == "object":
                    options = (
                        train_df[column]
                        .dropna()
                        .unique()
                        .tolist()
                    )
                    input_data[column] = st.selectbox(column, options)
                else:
                    median_value = float(
                        train_df[column]
                        .median()
                    )
                    input_data[column] = st.number_input(column, value=median_value)

    # --- Prediction Button and Result ---
    st.markdown("--- ")
    if st.button("Predict Purchase"):
        input_df = pd.DataFrame(
            [input_data]
        )

        prediction = model.predict(
            input_df
        )[0]

        probability = (
            model
            .predict_proba(
                input_df
            )[0][1]
        )

        st.subheader("Prediction Result")

        if prediction == 1:
            st.success(
                "Customer is likely to purchase."
            )
        else:
            st.warning(
                "Customer is unlikely to purchase."
            )

        st.metric("Purchase Probability", f"{probability:.2%}")
else:
    st.info("Please ensure the model and training data are available to run the prediction form.")
