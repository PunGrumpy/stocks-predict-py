from src.data.data_preparation import fetch_data, preprocess_data
from src.features.feature_engineering import create_features
from src.models.train_model import train_model, save_model
from src.models.predict_model import load_model, make_prediction

# Fetch and preprocess data
df = fetch_data('AAPL', '2020-01-01', '2023-01-01')
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
