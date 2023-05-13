import pickle
import numpy as np


def load_model(path):
    with open(path, "rb") as f:
        loaded_model = pickle.load(f)
    return loaded_model


def make_prediction(model, X):
    prediction = model.predict(X)
    return prediction
