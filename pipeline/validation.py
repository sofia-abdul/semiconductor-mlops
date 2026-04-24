import pandas as pd
from pipeline.config import DATA_PATH, TARGET_COLUMN


def validate_data() -> pd.DataFrame:
    """Validate dataset structure and integrity"""

    df = pd.read_csv(DATA_PATH)

    print("Dataset loaded successfully")
    print("Shape:", df.shape)

    
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")

    print("\nMissing values:")
    print(df.isnull().sum())

    if "fin_width_nm" in df.columns:
        nulls = df["fin_width_nm"].isnull().sum()
        print(f"\nfin_width_nm null values: {nulls}")

    print("\nValidation complete")
    return df


if __name__ == "__main__":
    validate_data()