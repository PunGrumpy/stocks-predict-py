def create_features(df):
    # Fill in feature engineering steps here
    X = df.drop("Close", axis=1)
    y = df["Close"]
    return X, y
