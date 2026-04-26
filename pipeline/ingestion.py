import pandas as pd

from pipeline.config import DATA_PATH, TABLE_NAME, RAW_TARGET_COLUMN, TARGET_COLUMN
from pipeline.db import get_engine


def ingest_data() -> pd.DataFrame:
    """
    Ingest SECOM dataset into MariaDB.

    Steps:
    1. Load raw CSV
    2. Rename and transform target column
    3. Store data in database
    """

    # Extract
    df = pd.read_csv(DATA_PATH)

    print("Raw data loaded")
    print("Shape:", df.shape)

    # Transform
    df = df.rename(columns={RAW_TARGET_COLUMN: TARGET_COLUMN})

    df[TARGET_COLUMN] = df[TARGET_COLUMN].map({-1: 0, 1: 1})

    print("\nTarget distribution after transformation:")
    print(df[TARGET_COLUMN].value_counts())

    # Load
    engine = get_engine()

    df.to_sql(
        TABLE_NAME,
        con=engine,
        if_exists="replace",
        index=False
    )

    print("\nIngestion complete")
    print(f"Rows loaded: {len(df)}")
    print(f"Columns stored: {len(df.columns)}")
    print(f"Table: {TABLE_NAME}")

    return df


if __name__ == "__main__":
    ingest_data()