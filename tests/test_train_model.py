import pandas as pd
from src.models.train_model import train_model


def test_train_model():
    X = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
    y = pd.Series([7, 8, 9])
    model = train_model(X, y)
    assert model is not None
