import pandas as pd

from pipeline.ingestion import ingest_data
from pipeline.config import TARGET_COLUMN


def test_ingestion_output():
    df = ingest_data()

    # Check dataframe not empty
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0

    # Check target column exists
    assert TARGET_COLUMN in df.columns

    # Check target values are binary
    assert set(df[TARGET_COLUMN].unique()).issubset({0, 1})