import pandas as pd

from pipeline.config import TARGET_COLUMN
from pipeline.preprocessing import preprocess_data


def test_preprocessing_output():
    processed_df = preprocess_data()

    assert isinstance(processed_df, pd.DataFrame)
    assert len(processed_df) > 0

    assert TARGET_COLUMN in processed_df.columns
    assert processed_df[TARGET_COLUMN].isnull().sum() == 0

    assert processed_df.isnull().sum().sum() == 0

    feature_columns = processed_df.drop(columns=[TARGET_COLUMN]).columns
    assert len(feature_columns) > 0

    non_numeric_columns = processed_df[feature_columns].select_dtypes(exclude=["int64", "float64"]).columns
    assert len(non_numeric_columns) == 0