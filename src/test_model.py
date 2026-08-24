

import os
import joblib

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "tourism_purchase_model.joblib"



def test_model_exists():

    assert os.path.exists(
        "/content/drive/My Drive/Colab_Notebooks/tourism_package_prediction/src/tourism_purchase_model.joblib"
    )


def test_model_loads():

    model = joblib.load(MODEL_PATH)

    assert model is not None
