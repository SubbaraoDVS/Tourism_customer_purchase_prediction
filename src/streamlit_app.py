
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# ============================================================
# 1. STREAMLIT PAGE CONFIGURATION
# IMPORTANT: This must be the first Streamlit command
# ============================================================

st.set_page_config(
    page_title="Tourism Purchase Prediction",
    page_icon="✈️",
    layout="wide"
)


# ============================================================
# 2. PROJECT CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "tourism_purchase_model.joblib"
TRAIN_DATA_PATH = "/content/drive/My Drive/Colab_Notebooks/tourism_package_prediction/artifacts/train.csv"
TARGET_COLUMN = "ProdTaken"


# ============================================================
# 3. LOAD TRAINED MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = joblib.load(MODEL_PATH)

    return model


# ============================================================
# 4. LOAD TRAINING DATA
# Used to dynamically create Streamlit input fields
# ============================================================

@st.cache_data
def load_training_data():

    train_df = pd.read_csv(TRAIN_DATA_PATH)

    return train_df


# ============================================================
# 5. APPLICATION TITLE
# ============================================================

st.title("✈️ Tourism Package Purchase Prediction")

st.write(
    """
    Enter customer information below to predict whether
    the customer is likely to purchase a tourism package.
    """
)

st.divider()


# ============================================================
# 6. CHECK REQUIRED FILES
# ============================================================

if not Path(MODEL_PATH).exists():

    st.error(
        f"""
        Model file not found:

        {MODEL_PATH}

        Please train the model before running the Streamlit app.
        """
    )

    st.stop()


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
# 7. LOAD MODEL AND DATA
# ============================================================

model = load_model()

train_df = load_training_data()


# ============================================================
# 8. IDENTIFY FEATURE COLUMNS
# ============================================================

feature_columns = [

    column

    for column in train_df.columns

    if column != TARGET_COLUMN

]


# ============================================================
# 9. CREATE CUSTOMER INPUT FORM
# ============================================================

st.subheader("Customer Information")


input_data = {}


with st.form("prediction_form"):

    # Create two columns for better UI
    col1, col2 = st.columns(2)


    for index, column in enumerate(feature_columns):

        # Alternate between left and right column
        current_column = col1 if index % 2 == 0 else col2


        with current_column:


            # ====================================================
            # CATEGORICAL VARIABLES
            # ====================================================

            if (
                train_df[column].dtype == "object"
                or str(train_df[column].dtype) == "category"
            ):

                options = (

                    train_df[column]
                    .dropna()
                    .unique()
                    .tolist()

                )

                input_data[column] = st.selectbox(

                    label=column,

                    options=options

                )


            # ====================================================
            # NUMERICAL VARIABLES
            # ====================================================

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


                input_data[column] = st.number_input(

                    label=column,

                    min_value=min_value,

                    max_value=max_value,

                    value=median_value

                )


    # ============================================================
    # PREDICTION BUTTON
    # ============================================================

    submitted = st.form_submit_button(
        "🔮 Predict Purchase"
    )


# ============================================================
# 10. MAKE PREDICTION
# ============================================================

if submitted:

    try:

        # Convert user inputs into DataFrame
        input_df = pd.DataFrame(
            [input_data]
        )


        # Ensure column order matches training data
        input_df = input_df[
            feature_columns
        ]


        # Generate prediction
        prediction = model.predict(
            input_df
        )[0]


        # Generate probability
        probability = (

            model
            .predict_proba(
                input_df
            )[0][1]

        )


        # ========================================================
        # 11. DISPLAY RESULT
        # ========================================================

        st.divider()

        st.subheader(
            "Prediction Result"
        )


        result_col1, result_col2 = st.columns(2)


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

                label="Purchase Probability",

                value=f"{probability:.2%}"

            )


        # ========================================================
        # 12. BUSINESS RECOMMENDATION
        # ========================================================

        st.subheader(
            "Recommended Business Action"
        )


        if probability >= 0.70:

            st.success(
                """
                HIGH-POTENTIAL CUSTOMER

                Recommended action:
                Prioritize this customer for immediate sales follow-up,
                personalized offers, and direct communication.
                """
            )


        elif probability >= 0.40:

            st.info(
                """
                MEDIUM-POTENTIAL CUSTOMER

                Recommended action:
                Use personalized email campaigns, promotional offers,
                and follow-up marketing.
                """
            )


        else:

            st.warning(
                """
                LOW-POTENTIAL CUSTOMER

                Recommended action:
                Use low-cost digital marketing campaigns and avoid
                allocating expensive sales resources.
                """
            )


        # ========================================================
        # 13. SHOW CUSTOMER INPUT DATA
        # ========================================================

        with st.expander(
            "View Customer Input Data"
        ):

            st.dataframe(
                input_df,
                use_container_width=True
            )


    except Exception as e:

        st.error(
            "An error occurred while making the prediction."
        )

        st.exception(e)
