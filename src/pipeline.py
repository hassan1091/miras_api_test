import os
import pandas as pd
import json
import joblib  # Import joblib for model persistence
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from utils.csv_to_json import read_old_factory_data_to_json

# Load input data from JSON file
def load_input_data(json_file):
    with open(json_file, 'r') as file:
        return json.load(file)

# Define a mapping for the feature names
feature_name_mapping = {
    'Temperature (آ°C)': 'Temperature (°C)',  # Add any other feature name mappings here
}

# Function to clean the input data by renaming columns to match training columns
def clean_input_data(input_data, feature_name_mapping):
    return {feature_name_mapping.get(k, k): v for k, v in input_data.items()}

# Sample data setup
data = read_old_factory_data_to_json()
df = pd.DataFrame(data)

# Encoding multiple target columns
target_columns = ['Device Status', 'Failure Mode', 'Issue Detected', 'Action Suggested']
label_encoders = {col: LabelEncoder() for col in target_columns}

# Apply label encoding for each target column
for col in target_columns:
    df[col] = label_encoders[col].fit_transform(df[col])

# Features and targets
X = df[['Temperature (°C)', 'Humidity (%)', 'Pressure (Pa)', 'Operating Hours', 'Response Time (ms)']]
y = df[target_columns]

# Split the dataset for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize SMOTE
smote = SMOTE(random_state=42, k_neighbors=2)

# Dictionary to store models for each target
models = {col: RandomForestClassifier(random_state=42) for col in target_columns}
y_pred = pd.DataFrame()

# Training and predicting for each target column
for col in target_columns:
    # Apply SMOTE to balance the training data for each target column
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train[col])

    # Train the model
    models[col].fit(X_train_smote, y_train_smote)

    # Save the model after training
    model_path = os.path.join(os.path.dirname(__file__), "..", 'data', 'models', f'{col}_model.pkl')

    joblib.dump(models[col],  model_path)  # Save the trained model

# Function to predict based on input data with weighted predictions
def predict(input_data):
    # Clean the input data by matching feature names
    cleaned_input_data = clean_input_data(input_data, feature_name_mapping)

    # Convert the cleaned input data into a DataFrame with the same structure as X
    input_df = pd.DataFrame([cleaned_input_data])

    # List to hold the final grouped predictions with their weights
    grouped_predictions = []

    # For each target column, get the predictions and the confidence (probabilities)
    all_predictions = {}

    for col in target_columns:
        # Load the saved model
        model_path = os.path.join(os.path.dirname(__file__), "..", 'data', 'models', f'{col}_model.pkl')
        models[col] = joblib.load(model_path)  # Load the trained model

        # Get predicted probabilities for each class (output of RandomForest)
        probabilities = models[col].predict_proba(input_df)[0]

        # Pair each class with its corresponding probability
        class_probabilities = [(label_encoders[col].classes_[i], probabilities[i]) for i in range(len(probabilities))]
        
        # Store sorted class probabilities for this column
        all_predictions[col] = sorted(class_probabilities, key=lambda x: x[1], reverse=True)

    # Iterate through all possible predictions and gather all attributes with their confidence
    # Create a single dictionary for all target columns, and append it to the results
    all_combined_predictions = {}

    # Pick the best (most likely) class for each target column and add it to a combined prediction
    for i in range(len(all_predictions[target_columns[0]])):  # Only need to iterate for one target as all are sorted
        combined_prediction = {}
        total_weight = 0  # Sum of all probabilities for normalization

        # Aggregate the prediction with the highest probability for each target
        for col in target_columns:
            combined_prediction[col] = all_predictions[col][i][0]
            total_weight += all_predictions[col][i][1]

        # Normalize the weight for this combined prediction
        normalized_weight = total_weight / len(target_columns)
        grouped_predictions.append([combined_prediction, normalized_weight])

    return grouped_predictions

# Example: Load real input data from JSON file
input_data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'input_data.json')

input_data = load_input_data(input_data_path)

# Predict for the loaded input data
predictions = predict(input_data)

# Format the output
output = {
    "input": input_data,
    "predictions": predictions
}

# Save output to a JSON file
predictions_output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'predictions_output.json')

with open(predictions_output_path, 'w') as output_file:
    json.dump(output, output_file, indent=4)

print(json.dumps(output, indent=4))  # Optional: print output to the console for verification
