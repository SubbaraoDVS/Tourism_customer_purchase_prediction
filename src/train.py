
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

# Define the base path for accessing files within the Colab environment
BASE_PROJECT_PATH = "/content/drive/My Drive/Colab_Notebooks/tourism_package_prediction"

def train_model():

    train_df = pd.read_csv(
        Path(BASE_PROJECT_PATH) / "artifacts" / "train.csv"
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
        Path(BASE_PROJECT_PATH) / "artifacts" / "mlruns"
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

        # Save the model to the src directory as specified by deployment instructions
        model_save_path = Path(BASE_PROJECT_PATH) / "src" / "tourism_purchase_model.joblib"
        model_save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            pipeline,
            model_save_path
        )

        print(
            "Model training completed."
        )


if __name__ == "__main__":

    train_model()
