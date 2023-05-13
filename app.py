import streamlit as st
from src.data.data_preparation import fetch_data, preprocess_data
from src.features.feature_engineering import create_features
from src.models.train_model import train_model, save_model
from src.models.predict_model import load_model, make_prediction
from datetime import date, timedelta

# Mapping of stock symbols to full names and currencies
stocks = {
    "AAPL": {"name": "Apple Inc.", "currency": "USD"},
    "GOOG": {"name": "Alphabet Inc. (Google)", "currency": "USD"},
    "MSFT": {"name": "Microsoft Corporation", "currency": "USD"},
    "AMZN": {"name": "Amazon.com, Inc.", "currency": "USD"},
    "FB": {"name": "Facebook, Inc.", "currency": "USD"},
    "PTT.BK": {"name": "PTT Public Company Limited", "currency": "THB"},
    "SCB.BK": {
        "name": "The Siam Commercial Bank Public Company Limited",
        "currency": "THB",
    },
    "BBL.BK": {"name": "Bangkok Bank Public Company Limited", "currency": "THB"},
}

# Ask user for stock symbol
ticker = st.selectbox("Select a stock symbol:", list(stocks.keys()))

# Calculate the start date as 30 days ago
start_date = date.today() - timedelta(days=30)

# Fetch and preprocess data
df = fetch_data(ticker, start_date)
df = preprocess_data(df)

# Feature engineering
X, y = create_features(df)

# Check if there is enough data
if len(X) < 10:  # adjust this number as necessary
    st.error("Not enough data to train the model.")
else:
    # Train the model
    model = train_model(X, y)

    # Save the model
    save_model(model, "models/model.pkl")

    # Load the model
    loaded_model = load_model("models/model.pkl")

    # Make prediction
    prediction = make_prediction(loaded_model, X[-1:])

    # Streamlit interface
    st.title(f'Stock Price Prediction for {stocks[ticker]["name"]}')
    st.line_chart(df["Close"])
    st.write(
        f'The predicted closing price for the next day is: {round(prediction[0], 2)} {stocks[ticker]["currency"]}'
    )
