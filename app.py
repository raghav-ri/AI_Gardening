import requests
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from weather_api import get_weather  # Import weather function

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enables cross-origin requests

# Load trained model, scaler, label encoders, and target encoder
with open("crop_model.pkl", "rb") as file:
    model, scaler, label_encoders, target_encoder = pickle.load(file)

@app.route('/get-crop', methods=['GET'])
def get_crop():
    """API Endpoint to get crop recommendation based on city weather"""
    city = request.args.get('city')

    if not city:
        return jsonify({"error": "City not provided"}), 400

    # Fetch weather data
    temperature, humidity, rainfall = get_weather(city)

    if temperature is None or humidity is None or rainfall is None:
        return jsonify({"error": "Failed to fetch weather data"}), 500

    # Prepare input data with default values for pH, soil type, and season
    input_data = pd.DataFrame([[temperature, humidity, rainfall, 6.5, "Loamy", "Summer"]],
                              columns=['temperature', 'humidity', 'rainfall', 'ph', 'soil_type', 'season'])
    
    # Encode categorical features
    for col in ['soil_type', 'season']:
        if col in label_encoders:
            input_data[col] = label_encoders[col].transform(input_data[col])
        else:
            print(f"Label encoder missing for {col}")
    
    # Scale numerical features
    input_scaled = scaler.transform(input_data)

    # Predict crop
    predicted_crop_encoded = model.predict(input_scaled)[0]

    # Ensure the predicted value is within target_encoder's range
    if predicted_crop_encoded not in range(len(target_encoder.classes_)):
        return jsonify({"recommended_crop": "Unknown Crop"})

    predicted_crop = target_encoder.inverse_transform([predicted_crop_encoded])[0]

    return jsonify({
        "city": city,
        "temperature": temperature,
        "humidity": humidity,
        "rainfall": rainfall,
        "recommended_crop": predicted_crop
    })

# Note: Do not include app.run() for production. Deployment platform uses gunicorn.
