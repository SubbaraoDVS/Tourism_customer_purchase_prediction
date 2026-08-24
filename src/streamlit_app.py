
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# MUST BE THE FIRST STREAMLIT COMMAND
# ============================================================

st.set_page_config(
    page_title="Tourism Purchase Prediction",
    page_icon="✈️",
    layout="wide"
)


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_PATH = (
    "/content/drive/My Drive/"
    "Colab_Notebooks/"
    "tourism_package_prediction"
)


# ============================================================
# FILE PATHS
# ============================================================

MODEL_PATH = (
    f"{PROJECT_PATH}/models/"
    "tourism_purchase_model.joblib"
)

TRAIN_DATA_PATH = (
    f"{PROJECT_PATH}/artifacts/"
    "train.csv"
)

TARGET_COLUMN = "ProdTaken"


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    return joblib.load(
        MODEL_PATH
    )


# ============================================================
# LOAD TRAINING DATA
# ============================================================

@st.cache_data
def load_training_data():

    return pd.read_csv(
        TRAIN_DATA_PATH
    )


# ============================================================
# TITLE
# ============================================================

st.title(
    "✈️ Tourism Package Purchase Prediction"
)


# ============================================================
# CHECK MODEL
# ============================================================

if not Path(MODEL_PATH).exists():

    st.error(
        f"Model file not found:\n\n{MODEL_PATH}"
    )

    st.stop()


# ============================================================
# CHECK TRAIN DATA
# ============================================================

if not Path(TRAIN_DATA_PATH).exists():

    st.error(
        f"""
Training data file not found:

{TRAIN_DATA_PATH}

Please run the data preparation step first.
"""
    )

    st.stop()


# ============================================================
# LOAD FILES
# ============================================================

model = load_model()

train_df = load_training_data()


# ============================================================
# FEATURE COLUMNS
# ============================================================

feature_columns = [

    column
    for column in train_df.columns
    if column != TARGET_COLUMN

]


# ============================================================
# CUSTOMER INPUT FORM
# ============================================================

st.subheader(
    "Enter Customer Information"
)

input_data = {}


with st.form("prediction_form"):

    col1, col2 = st.columns(2)


    for index, column in enumerate(feature_columns):

        current_column = (
            col1
            if index % 2 == 0
            else col2
        )


        with current_column:

            # CATEGORICAL FEATURES
            if (
                train_df[column].dtype == "object"
                or str(train_df[column].dtype)
                == "category"
            ):

                options = (
                    train_df[column]
                    .dropna()
                    .unique()
                    .tolist()
                )

                input_data[column] = (
                    st.selectbox(
                        column,
                        options
                    )
                )


            # NUMERICAL FEATURES
            else:

                median_value = float(
                    train_df[column]
                    .median()
                )

                min_value = float(
                    train_df[column]
                    .min()
                )

                max_value = float(
                    train_df[column]
                    .max()
                )

                input_data[column] = (
                    st.number_input(
                        column,
                        min_value=min_value,
                        max_value=max_value,
                        value=median_value
                    )
                )


    submitted = st.form_submit_button(
        "🔮 Predict Purchase"
    )


# ============================================================
# PREDICTION
# ============================================================

if submitted:

    try:

        input_df = pd.DataFrame(
            [input_data]
        )

        # Ensure feature order
        input_df = input_df[
            feature_columns
        ]


        prediction = model.predict(
            input_df
        )[0]


        probability = (
            model
            .predict_proba(
                input_df
            )[0][1]
        )


        # ====================================================
        # DISPLAY RESULT
        # ====================================================

        st.divider()

        st.subheader(
            "Prediction Result"
        )


        result_col1, result_col2 = (
            st.columns(2)
        )


        with result_col1:

            if prediction == 1:

                st.success(
                    "✅ Customer is likely to purchase the tourism package."
                )

            else:

                st.warning(
                    "⚠️ Customer is unlikely to purchase the tourism package."
                )


        with result_col2:

            st.metric(
                "Purchase Probability",
                f"{probability:.2%}"
            )


        # ====================================================
        # BUSINESS RECOMMENDATION
        # ====================================================

        st.subheader(
            "Recommended Business Action"
        )


        if probability >= 0.70:

            st.success(
                """
HIGH-POTENTIAL CUSTOMER

Recommended Action:
Prioritize this customer for immediate
sales follow-up and personalized offers.
"""
            )


        elif probability >= 0.40:

            st.info(
                """
MEDIUM-POTENTIAL CUSTOMER

Recommended Action:
Use personalized marketing campaigns,
promotional offers, and follow-ups.
"""
            )


        else:

            st.warning(
                """
LOW-POTENTIAL CUSTOMER

Recommended Action:
Use low-cost digital marketing campaigns.
"""
            )


    except Exception as e:

        st.error(
            "Prediction failed."
        )

        st.exception(e)
