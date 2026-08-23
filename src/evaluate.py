
import pandas as pd
import joblib
import json

from pathlib import Path

from sklearn.metrics import (

    accuracy_score,

    precision_score,

    recall_score,

    f1_score,

    roc_auc_score
)


TARGET_COLUMN = "ProdTaken"

MIN_F1_SCORE = 0.60

# Define the base path for accessing files within the Colab environment
BASE_PROJECT_PATH = "/content/drive/My Drive/Colab_Notebooks/tourism_package_prediction"

def evaluate_model():

    test_df = pd.read_csv(
        Path(BASE_PROJECT_PATH) / "artifacts" / "test.csv"
    )

    X_test = test_df.drop(
        columns=[TARGET_COLUMN]
    )

    y_test = test_df[
        TARGET_COLUMN
    ]

    # Load the model from the src directory
    model_load_path = Path(BASE_PROJECT_PATH) / "src" / "tourism_purchase_model.joblib"
    model = joblib.load(model_load_path)

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

    # Save metrics to artifacts directory
    metrics_save_path = Path(BASE_PROJECT_PATH) / "artifacts" / "model_metrics.json"
    metrics_save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(
        metrics_save_path,
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
