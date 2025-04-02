from data_preprocessing import preprocess_data
from imblearn.over_sampling import SMOTE
import pandas as pd

# Step 1: Preprocess the dataset
df = preprocess_data("Crop_recommendation_expanded.csv")  # Change this to your actual dataset file

# Step 2: Separate features (X) and target (y)
X = df.iloc[:, :-1]  # Select all columns except the last one as features
y = df.iloc[:, -1]   # Select the last column as the target variable

# Step 3: Convert categorical variables into numeric using one-hot encoding
X = pd.get_dummies(X)

# Step 4: Apply SMOTE to handle class imbalance
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Step 5: Convert resampled data into a DataFrame
df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
df_resampled['Target'] = y_resampled  # Add the target column back

# Step 6: Save the processed dataset to a CSV file
df_resampled.to_csv("processed_dataset.csv", index=False)

print("Data preprocessing and SMOTE balancing completed successfully!")
