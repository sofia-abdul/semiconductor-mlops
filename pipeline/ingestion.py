import pandas as pd

from pipeline.config import DATA_PATH, DROP_COLUMNS, TABLE_NAME
from pipeline.db import get_engine


def ingest_data() -> pd.DataFrame:
    """
    Ingest semiconductor dataset into MariaDB.

    Steps:
    1. Load raw CSV
    2. Apply basic transformations (drop non-informative columns)
    3. Store data in database
    """

    # Extract
    df = pd.read_csv(DATA_PATH)

    # Transform
    columns_to_drop = [col for col in DROP_COLUMNS if col in df.columns]
    df = df.drop(columns=columns_to_drop)

    # Load
    engine = get_engine()

    df.to_sql(
        TABLE_NAME,
        con=engine,
        if_exists="replace",   #  repeatable pipeline
        index=False
    )

    print("Ingestion complete")
    print(f"Rows loaded: {len(df)}")
    print(f"Columns stored: {len(df.columns)}")

    return df


if __name__ == "__main__":
    ingest_data()