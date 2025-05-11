# Import necessary libraries for data handling and machine learning
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
# We collected sensor data (accelerometer and gyroscope readings) and stored it in a CSV file.
file_path = '/content/mcw.csv'
data = pd.read_csv(file_path)  # Read the CSV into a pandas DataFrame

# Step 1: Summarize data into 6 features (mean values of accelerometer and gyroscope readings)
# This function helps us convert raw multi-sample sensor data into a single row of summarized features for each event.
def summarize_data(df):
    summary_df = pd.DataFrame()

    # We computed the average of all 'acc_x' columns for each row
    summary_df['acc_x_mean'] = df.filter(regex='acc_x').mean(axis=1)
    summary_df['acc_y_mean'] = df.filter(regex='acc_y').mean(axis=1)
    summary_df['acc_z_mean'] = df.filter(regex='acc_z').mean(axis=1)

    # We also computed the average of all 'gy_x' columns for each row
    summary_df['gy_x_mean'] = df.filter(regex='gy_x').mean(axis=1)
    summary_df['gy_y_mean'] = df.filter(regex='gy_y').mean(axis=1)
    summary_df['gy_z_mean'] = df.filter(regex='gy_z').mean(axis=1)

    # Copy the label column (e.g., "fall" or "no fall") to the summary
    summary_df['label'] = df['label']
    return summary_df

# Summarize the dataset
# We now have 6 key features for each data instance, which simplifies training.
summary_data = summarize_data(data)

# Separate features and target label
# 'X' contains the features, 'y' contains the target labels ("fall" or "no fall")
X = summary_data.drop(columns=['label'])
y = summary_data['label']

# Encode labels into numerical values
# Since ML models work with numbers, we converted text labels into numeric classes.
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split dataset into training and testing sets
# We used 80% of the data for training and 20% for testing to evaluate performance.
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train a RandomForest Classifier
# We chose RandomForest for its accuracy and ability to handle tabular feature data well.
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)  # Fit the model to training data

# Evaluate the model
# We tested the trained model on the 20% test data
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

# Print the model's performance
print("Model Accuracy:", accuracy)
print("Classification Report:\n", report)

# Function to classify a single input
# We created this function so we can classify new sensor samples (summarized as 6 values).
def classify_fall_single(input_data):
    """
    Classifies a single set of accelerometer and gyroscope values.

    Parameters:
        input_data (list or array): A list of 6 values [acc_x_mean, acc_y_mean, acc_z_mean, gy_x_mean, gy_y_mean, gy_z_mean].

    Returns:
        str: Classification result ('fall' or 'no fall').
    """
    # Validate input size
    if len(input_data) != 6:
        raise ValueError("Input data must contain exactly 6 values: [acc_x_mean, acc_y_mean, acc_z_mean, gy_x_mean, gy_y_mean, gy_z_mean].")

    # Reshape the input data and make a prediction using our trained model
    input_data_array = np.array(input_data).reshape(1, -1)
    prediction_encoded = model.predict(input_data_array)

    # Convert numeric prediction back to original label
    prediction_label = label_encoder.inverse_transform(prediction_encoded)

    # Return the predicted class label
    return prediction_label[0]

# Example usage
# We tested our classifier with a dummy sample input (you can replace this with actual data from sensors).
sample_input = [0.1, -0.3, 9.8, 0.02, 0.01, -0.03]  # Replace with a new 6-value sample
predicted_label = classify_fall_single(sample_input)
print("Predicted Label:", predicted_label)
