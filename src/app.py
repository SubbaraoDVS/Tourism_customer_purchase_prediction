
import streamlit as st
import pandas as pd
import joblib
import os

# Define the base path for accessing files within the Colab environment
# This path assumes the project root is '/content/drive/My Drive/Colab_Notebooks/tourism_package_prediction'
# and the Streamlit app (app.py) is in the 'src/' subdirectory.
BASE_PROJECT_PATH = "/content/drive/My Drive/Colab_Notebooks/tourism_package_prediction"
MODEL_PATH = os.path.join(BASE_PROJECT_PATH, "src", "tourism_purchase_model.joblib")
TRAIN_DATA_PATH = os.path.join(BASE_PROJECT_PATH, "artifacts", "train.csv")

@st.cache_resource
def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except FileNotFoundError:
        st.error(f"Model file not found at: {MODEL_PATH}. Please ensure the model is trained and saved.")
        return None

@st.cache_data
def load_train_data():
    try:
        return pd.read_csv(TRAIN_DATA_PATH)
    except FileNotFoundError:
        st.error(f"Training data file not found at: {TRAIN_DATA_PATH}. Cannot infer feature types.")
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
    if st.button("Predict Purchase"): # Moved button outside columns
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
