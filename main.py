import streamlit as st
from src.data.data_preparation import fetch_data, preprocess_data
from src.features.feature_engineering import create_features
from src.models.train_model import train_model, save_model
from src.models.predict_model import load_model, make_prediction

# Ask user for stock symbol
ticker = st.text_input("Enter a stock symbol (e.g. AAPL for Apple, PTT.BK for PTT):", value="AAPL")

# Fetch and preprocess data
df = fetch_data(ticker, '2020-01-01', '2023-01-01')
df = preprocess_data(df)

# Feature engineering
X, y = create_features(df)

# Train the model
model = train_model(X, y)

# Save the model
save_model(model, 'models/model.pkl')

# Load the model
loaded_model = load_model('models/model.pkl')

# Make prediction
prediction = make_prediction(loaded_model, X[-1:])

# Streamlit interface
st.title('Stock Price Prediction')
st.line_chart(df['Close'])
st.write('The predicted closing price for the next day is: $', prediction[0])
