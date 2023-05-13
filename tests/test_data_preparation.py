import pandas as pd
from src.data.data_preparation import fetch_data, preprocess_data


def test_fetch_data():
    df = fetch_data("AAPL", "2020-01-01", "2020-01-31")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_preprocess_data():
    df = pd.DataFrame({"Close": [1, 2, 3]})
    processed_df = preprocess_data(df)
    assert isinstance(processed_df, pd.DataFrame)
