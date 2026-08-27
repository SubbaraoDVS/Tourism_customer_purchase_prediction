
import pandas as pd
import json
from pathlib import Path


DATA_PATH = "/content/drive/My Drive/Colab_Notebooks/tourism_package_prediction/data/tourism.csv"
TARGET_COLUMN = "ProdTaken"


def validate_data(
    input_path=DATA_PATH,
    target_column=TARGET_COLUMN
):

    df = pd.read_csv(input_path)

    validation_report = {

        "rows": int(df.shape[0]),

        "columns": int(df.shape[1]),

        "duplicate_rows": int(
            df.duplicated().sum()
        ),

        "missing_values": (
            df.isnull()
            .sum()
            .to_dict()
        ),

        "target_exists": (
            target_column in df.columns
        )
    }

    Path("artifacts").mkdir(
        exist_ok=True
    )

    with open(
        "/content/drive/My Drive/Colab_Notebooks/tourism_package_prediction/artifacts/validation_report.json",
        "w"
    ) as file:

        json.dump(
            validation_report,
            file,
            indent=4
        )

    if not validation_report["target_exists"]:

        raise ValueError(
            f"Target column "
            f"'{target_column}' "
            f"not found."
        )

    print("Data validation successful.")

    return validation_report


if __name__ == "__main__":

    report = validate_data()

    print(report)

    print("Data validation completed.")

    return report


if __name__ == "__main__":

    report = validate_data(
        "/content/drive/My Drive/Colab_Notebooks/tourism_package_prediction/data/tourism.csv"
    )

    print(report)
