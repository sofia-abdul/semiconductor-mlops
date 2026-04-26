import pandas as pd

from sklearn.preprocessing import StandardScaler

from pipeline.config import TABLE_NAME, TARGET_COLUMN, PROCESSED_DATA_PATH
from pipeline.db import get_engine


MISSING_THRESHOLD = 50.0
LOW_VARIANCE_THRESHOLD = 1e-5


def load_ingested_data() -> pd.DataFrame:
    """Load ingested SECOM data from MariaDB."""

    engine = get_engine()
    df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", con=engine)

    print("Loaded SECOM data from MariaDB")
    print("Shape before preprocessing:", df.shape)

    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Drop high-missing columns and impute remaining missing values."""

    df = df.copy()

    missing_percent = df.isnull().mean() * 100
    columns_to_drop = missing_percent[missing_percent > MISSING_THRESHOLD].index.tolist()

    df = df.drop(columns=columns_to_drop)

    print(f"\nDropped columns with >{MISSING_THRESHOLD}% missing values: {len(columns_to_drop)}")

    remaining_missing_before = df.isnull().sum().sum()
    print("Remaining missing values before imputation:", remaining_missing_before)

    numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    if TARGET_COLUMN in numeric_columns:
        numeric_columns.remove(TARGET_COLUMN)

    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())

    remaining_missing_after = df.isnull().sum().sum()
    print("Remaining missing values after imputation:", remaining_missing_after)

    return df


def remove_low_variance_features(df: pd.DataFrame) -> pd.DataFrame:
    """Remove features with near-zero variance."""

    df = df.copy()

    numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    if TARGET_COLUMN in numeric_columns:
        numeric_columns.remove(TARGET_COLUMN)

    variance = df[numeric_columns].var()
    low_variance_columns = variance[variance < LOW_VARIANCE_THRESHOLD].index.tolist()

    df = df.drop(columns=low_variance_columns)

    print(f"\nRemoved low-variance features: {len(low_variance_columns)}")

    return df


def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    """Scale numerical features and keep target unchanged."""

    df = df.copy()

    y = df[TARGET_COLUMN]
    X = df.drop(columns=[TARGET_COLUMN])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    processed_df = pd.DataFrame(X_scaled, columns=X.columns)
    processed_df[TARGET_COLUMN] = y.values

    print("\nFeatures scaled using StandardScaler")
    print("Shape after preprocessing:", processed_df.shape)

    return processed_df


def preprocess_data() -> pd.DataFrame:
    """Run the full SECOM preprocessing stage."""

    df = load_ingested_data()

    df = handle_missing_values(df)
    df = remove_low_variance_features(df)
    processed_df = scale_features(df)

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(PROCESSED_DATA_PATH, index=False)

    print("\nPreprocessing complete")
    print(f"Processed data saved to: {PROCESSED_DATA_PATH}")

    return processed_df


if __name__ == "__main__":
    preprocess_data()