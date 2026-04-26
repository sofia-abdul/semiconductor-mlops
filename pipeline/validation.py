import pandas as pd

from pipeline.config import DATA_PATH


EXPECTED_TARGET_COLUMN = "Pass/Fail"


def validate_data() -> pd.DataFrame:
    """
    Validate the raw SECOM dataset before ingestion.
    """

    df = pd.read_csv(DATA_PATH)

    print("SECOM dataset loaded successfully")
    print("Shape:", df.shape)

    if EXPECTED_TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column: {EXPECTED_TARGET_COLUMN}")

    print("\nTarget values:")
    print(df[EXPECTED_TARGET_COLUMN].value_counts())

    expected_labels = {-1, 1}
    actual_labels = set(df[EXPECTED_TARGET_COLUMN].dropna().unique())

    if actual_labels != expected_labels:
        raise ValueError(
            f"Unexpected target labels found: {actual_labels}. "
            f"Expected labels: {expected_labels}"
        )

    missing_values = df.isnull().sum()
    total_missing = missing_values.sum()

    print("\nTotal missing values:", total_missing)
    print("\nColumns with highest missing values:")
    print(missing_values.sort_values(ascending=False).head(10))

    duplicate_rows = df.duplicated().sum()
    print("\nDuplicate rows:", duplicate_rows)

    print("\nValidation complete")
    return df


if __name__ == "__main__":
    validate_data()