import pandas as pd

from sklearn.preprocessing import StandardScaler

from pipeline.config import TABLE_NAME, TARGET_COLUMN, PROCESSED_DATA_PATH
from pipeline.db import get_engine


def load_ingested_data() -> pd.DataFrame:
    engine = get_engine()
    df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", con=engine)

    print("Loaded ingested data from MariaDB")
    print("Shape before preprocessing:", df.shape)

    return df


def inspect_data_quality(df: pd.DataFrame) -> None:
    print("\nMissing values before preprocessing:")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0])

    print("\nTarget distribution:")
    print(df[TARGET_COLUMN].value_counts())

    print("\nTarget distribution (%):")
    print((df[TARGET_COLUMN].value_counts(normalize=True) * 100).round(2))

    numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns
    print("\nNumerical summary:")
    print(df[numeric_columns].describe())


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "fin_width_nm" in df.columns:
        # Planar transistors do not have fins, so missing values represent absence.
        df["fin_width_nm"] = df["fin_width_nm"].fillna(0)
        print("\nfin_width_nm missing values replaced with 0")

    print("\nMissing values after preprocessing:")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0])

    return df


def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns
    numeric_columns = [col for col in numeric_columns if col != TARGET_COLUMN]

    for col in numeric_columns:
        lower_bound = df[col].quantile(0.01)
        upper_bound = df[col].quantile(0.99)
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

    print("\nOutliers capped using 1st and 99th percentiles")
    return df


def encode_and_scale(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    y = df[TARGET_COLUMN]
    X = df.drop(columns=[TARGET_COLUMN])

    categorical_columns = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_columns = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    X_encoded = pd.get_dummies(X, columns=categorical_columns, drop_first=True)

    scaler = StandardScaler()
    X_encoded[numeric_columns] = scaler.fit_transform(X_encoded[numeric_columns])

    processed_df = X_encoded.copy()
    processed_df[TARGET_COLUMN] = y.values

    print("\nCategorical columns encoded:", len(categorical_columns))
    print("Numerical columns scaled:", len(numeric_columns))
    print("Shape after preprocessing:", processed_df.shape)

    return processed_df


def preprocess_data() -> pd.DataFrame:
    df = load_ingested_data()
    inspect_data_quality(df)

    df = handle_missing_values(df)
    df = handle_outliers(df)

    processed_df = encode_and_scale(df)
    processed_df.to_csv(PROCESSED_DATA_PATH, index=False)

    print(f"\nProcessed dataset saved to: {PROCESSED_DATA_PATH}")

    return processed_df


if __name__ == "__main__":
    preprocess_data()