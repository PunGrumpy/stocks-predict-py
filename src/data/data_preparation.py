import pandas as pd
import yfinance as yf


def fetch_data(ticker, start_date=None, end_date=None):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data


def preprocess_data(data):
    data = data.dropna()
    return data
