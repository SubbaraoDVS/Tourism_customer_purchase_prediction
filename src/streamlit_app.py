
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "tourism_purchase_model.joblib"


@st.cache_resource
def load_model():

    return joblib.load(MODEL_PATH)


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
