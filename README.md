# Stock Price Prediction App 📈

This application predicts the closing price of a selected stock for the next trading day. It uses historical stock prices and a machine learning model to make its predictions.

The application is built using Python, Streamlit, scikit-learn, and yfinance (Yahoo Finance's market data downloader). The user interface allows you to select a stock from a pre-defined list and displays a chart of the historical closing prices as well as the predicted closing price for the next day.

## Project Structure 📂

The project is structured as follows:

`app.py`: The main Streamlit application.\
`src/`: Contains the source code for data preparation, feature engineering, model training, and model prediction.\
`models/`: Contains the trained machine learning model.\
`tests/`: Contains tests for the various components of the application.

## How to Run the Application 🏃‍♂️

1. Clone this repository.
2. Install the required Python packages. You can do this by running `pip install -r requirements.txt`.
3. Run the Streamlit application by executing `streamlit run app.py`.

## Contributing 🤝

Contributions are welcome! Please feel free to submit a Pull Request.
