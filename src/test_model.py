

import os
import joblib


def test_model_exists():

    assert os.path.exists(
        "/content/drive/My Drive/Colab_Notebooks/tourism_package_prediction/src/tourism_purchase_model.joblib"
    )


def test_model_loads():

    model = joblib.load(
        "/content/drive/My Drive/Colab_Notebooks/tourism_package_prediction/src/tourism_purchase_model.joblib"
    )

    assert model is not None
