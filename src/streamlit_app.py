
import streamlit as st
import pandas as pd
import joblib


@st.cache_resource
def load_model():

    return joblib.load(
        "/content/drive/My Drive/Colab_Notebooks/tourism_package_prediction/models/tourism_purchase_model.joblib"
    )


model = load_model()


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
