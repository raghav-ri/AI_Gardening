import pandas as pd
import pickle
import numpy as np

# ✅ Load trained model & scaler
with open("crop_model_expanded.pkl", "rb") as file:
    model, scaler, label_encoders, target_encoder = pickle.load(file)

# ✅ Load label mapping
label_mapping = pd.read_csv("Crop_Label_Mapping.csv")
label_dict = dict(zip(label_mapping["Encoded_Label"], label_mapping["Crop_Name"]))

# ✅ Function to Predict Crop
def predict_crop(temperature, humidity, rainfall, ph, soil_type, season):
    # Convert categorical inputs using stored encoders
    soil_type_encoded = label_encoders["soil_type"].transform([soil_type])[0]
    season_encoded = label_encoders["season"].transform([season])[0]

    # Prepare input data
    input_data = np.array([[temperature, humidity, rainfall, ph, soil_type_encoded, season_encoded]])
    input_scaled = scaler.transform(input_data)

    # Predict encoded label
    predicted_label = model.predict(input_scaled)[0]

    # Convert to crop name
    predicted_crop = label_dict.get(predicted_label, "Unknown Crop")
    
    return predicted_crop

# ✅ Example Usage
predicted_crop = predict_crop(25, 70, 200, 6.5, "Loamy", "Winter")  # Example input
print(f"🌱 Recommended Crop: {predicted_crop}")
