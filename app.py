import datetime
import streamlit as st
import plotly.graph_objects as go
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
    "KBANK.BK": {"name": "Kasikornbank Public Company Limited", "currency": "THB"},
}

# Ask user for stock symbol
ticker = st.selectbox("Select a stock symbol:", list(stocks.keys()))

# Streamlit interface
st.title(f'Stock Price Prediction for {stocks[ticker]["name"]}')

# Let the user select the number of days
days = st.slider(
    "Number of past days to consider:", min_value=15, max_value=120, value=30, step=5
)

# Calculate the start date as 30 days ago
start_date = date.today() - timedelta(days=days)

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
    model, mae, mse, r2, cv_score = train_model(X, y)

    # Save the model
    save_model(model, "models/model.pkl")

    # Load the model
    loaded_model = load_model("models/model.pkl")

    # Make prediction
    prediction = make_prediction(loaded_model, X[-1:])

    # Plot the actual values vs predictions using Plotly
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Close"],
            mode="lines",
            name="Actual values",
            line=dict(color="blue", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=loaded_model.predict(X),
            mode="lines",
            name="Predicted values",
            line=dict(color="red", width=1),
        )
    )

    fig.update_layout(
        title=f"Actual vs Predicted Stock Prices for {stocks[ticker]['name']}",
        xaxis_title="Date",
        yaxis_title=f"Price (in {stocks[ticker]['currency']})",
        legend_title="Legend",
        font=dict(family="Courier New, monospace", size=18, color="RebeccaPurple"),
    )

    st.plotly_chart(fig)

    st.write(
        f'The predicted closing price for the next day is: **{round(prediction[0], 2)} {stocks[ticker]["currency"]}**'
    )

    # Advise the user on whether to buy or sell
    current_price = df["Close"].iloc[-1]
    if prediction[0] > current_price:
        st.write(
            f"The predicted closing price is higher than the current closing price. This could be a good opportunity to buy {ticker}."
        )
    elif prediction[0] < current_price:
        st.write(
            f"The predicted closing price is lower than the current closing price. You might want to consider selling {ticker}."
        )
    else:
        st.write(
            f"The predicted closing price is equal to the current closing price. You might want to hold onto your {ticker} stocks or monitor the market further."
        )

    # Display performance metrics
    st.subheader("Model Performance Metrics:")
    st.markdown(f"* Mean Absolute Error (MAE): {mae:.2f}")
    st.markdown(f"* Mean Squared Error (MSE): {mse:.2f}")
    st.markdown(f"* R^2 Score: {r2:.2f}")
    st.markdown(f"* Cross-Validation Score: {cv_score:.2f}")
