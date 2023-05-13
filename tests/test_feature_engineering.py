import pandas as pd
from src.features.feature_engineering import create_features


def test_create_features():
    df = pd.DataFrame({"Close": [1, 2, 3]})
    X, y = create_features(df)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
