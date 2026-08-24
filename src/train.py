
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from pathlib import Path

from sklearn.compose import (
    ColumnTransformer
)

from sklearn.pipeline import (
    Pipeline
)

from sklearn.impute import (
    SimpleImputer
)

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from sklearn.ensemble import (
    RandomForestClassifier
)


TARGET_COLUMN = "ProdTaken"


def train_model():

    train_df = pd.read_csv(
        "/content/drive/My Drive/Colab_Notebooks/tourism_package_prediction/artifacts/train.csv"
    )

    X_train = train_df.drop(
        columns=[TARGET_COLUMN]
    )

    y_train = train_df[
        TARGET_COLUMN
    ]

    numerical_columns = (
        X_train
        .select_dtypes(
            include=["int64", "float64"]
        )
        .columns
        .tolist()
    )

    categorical_columns = (
        X_train
        .select_dtypes(
            include=["object", "category"]
        )
        .columns
        .tolist()
    )

    numerical_pipeline = Pipeline(

        steps=[

            (
                "imputer",

                SimpleImputer(
                    strategy="median"
                )
            ),

            (
                "scaler",

                StandardScaler()
            )
        ]
    )

    categorical_pipeline = Pipeline(

        steps=[

            (
                "imputer",

                SimpleImputer(
                    strategy="most_frequent"
                )
            ),

            (
                "encoder",

                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(

        transformers=[

            (
                "num",

                numerical_pipeline,

                numerical_columns
            ),

            (
                "cat",

                categorical_pipeline,

                categorical_columns
            )
        ]
    )

    model = RandomForestClassifier(

        n_estimators=200,

        random_state=42,

        n_jobs=-1
    )

    pipeline = Pipeline(

        steps=[

            (
                "preprocessor",

                preprocessor
            ),

            (
                "model",

                model
            )
        ]
    )

    mlflow.set_tracking_uri(
        "file:/content/drive/My Drive/Colab_Notebooks/tourism_package_prediction/articrafts/mlruns"
    )

    mlflow.set_experiment(
        "Tourism_Purchase_Prediction"
    )

    with mlflow.start_run():

        pipeline.fit(
            X_train,
            y_train
        )

        mlflow.log_params({

            "model":
                "RandomForestClassifier",

            "n_estimators":
                200,

            "random_state":
                42
        })

        mlflow.sklearn.log_model(
            pipeline,
            name="model"
        )

        Path("models").mkdir(
            exist_ok=True
        )

        joblib.dump(

            pipeline,

            "/content/drive/My Drive/Colab_Notebooks/tourism_package_prediction/src/tourism_purchase_model.joblib"
        )

        print(
            "Model training completed."
        )


if __name__ == "__main__":

    train_model()
