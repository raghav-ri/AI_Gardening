import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

with open("crop_model.pkl", "rb") as file:
    model, scaler, label_encoders, target_encoder = pickle.load(file)

print("Model Loaded Successfully!")
print("Target Classes:", target_encoder.classes_)


# Load the processed dataset
df = pd.read_csv("processed_dataset.csv")

# Separate features and target variable
X = df.iloc[:, :-1]  # Features (all columns except last)
y = df.iloc[:, -1]   # Target variable (last column)

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Print detailed classification report
print(classification_report(y_test, y_pred))

# Save the trained model
import joblib
joblib.dump(model, "seed_prediction_model.pkl")

print("Model training completed and saved successfully!")
