
import pandas as pd
import joblib
import json

from sklearn.metrics import (

    accuracy_score,

    precision_score,

    recall_score,

    f1_score,

    roc_auc_score
)


TARGET_COLUMN = "ProdTaken"

MIN_F1_SCORE = 0.60


def evaluate_model():

    test_df = pd.read_csv(
        "/content/drive/My Drive/Colab_Notebooks/tourism_package_prediction/artifacts/test.csv"
    )

    X_test = test_df.drop(
        columns=[TARGET_COLUMN]
    )

    y_test = test_df[
        TARGET_COLUMN
    ]

    model = joblib.load(
        "/content/drive/My Drive/Colab_Notebooks/tourism_package_prediction/models/tourism_purchase_model.joblib"
    )

    predictions = model.predict(
        X_test
    )

    probabilities = (

        model

        .predict_proba(
            X_test
        )

        [:, 1]
    )

    metrics = {

        "accuracy":

        float(
            accuracy_score(
                y_test,
                predictions
            )
        ),

        "precision":

        float(
            precision_score(
                y_test,
                predictions,
                zero_division=0
            )
        ),

        "recall":

        float(
            recall_score(
                y_test,
                predictions,
                zero_division=0
            )
        ),

        "f1_score":

        float(
            f1_score(
                y_test,
                predictions,
                zero_division=0
            )
        ),

        "roc_auc":

        float(
            roc_auc_score(
                y_test,
                probabilities
            )
        )
    }

    print(metrics)

    with open(
        "/content/drive/My Drive/Colab_Notebooks/tourism_package_prediction/artifacts/model_metrics.json",
        "w"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    if metrics["f1_score"] < MIN_F1_SCORE:

        raise ValueError(

            f"Model failed performance gate. "

            f"F1 Score: "

            f"{metrics['f1_score']:.4f}"

        )

    print(
        "Model passed performance gate."
    )

    return metrics


if __name__ == "__main__":

    evaluate_model()
