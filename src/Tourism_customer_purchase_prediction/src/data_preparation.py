
import pandas as pd

from pathlib import Path

from sklearn.model_selection import (
    train_test_split
)


DATA_PATH = "/content/drive/My Drive/Colab_Notebooks/tourism_package_prediction/data/tourism.csv"

TARGET_COLUMN = "ProdTaken"


def prepare_data():

    df = pd.read_csv(
        DATA_PATH
    )

    X = df.drop(
        columns=[TARGET_COLUMN]
    )

    y = df[
        TARGET_COLUMN
    ]

    X_train, X_test, y_train, y_test = (
        train_test_split(

            X,
            y,

            test_size=0.20,

            random_state=42,

            stratify=y
        )
    )

    train_df = X_train.copy()

    train_df[
        TARGET_COLUMN
    ] = y_train

    test_df = X_test.copy()

    test_df[
        TARGET_COLUMN
    ] = y_test

    Path("artifacts").mkdir(
        exist_ok=True
    )

    train_df.to_csv(
        "/content/drive/My Drive/Colab_Notebooks/tourism_package_prediction/artifacts/train.csv",
        index=False
    )

    test_df.to_csv(
        "/content/drive/My Drive/Colab_Notebooks/tourism_package_prediction/artifacts/test.csv",
        index=False
    )

    print("Data preparation completed.")

    print(
        "Training Shape:",
        train_df.shape
    )

    print(
        "Testing Shape:",
        test_df.shape
    )


if __name__ == "__main__":

    prepare_data()
