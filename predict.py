import joblib
import numpy as np
from feature import extract_features

# Load model
model = joblib.load("models/speaker_model.pkl")

# Test file (change this later)
file_path = "data/speaker1/speaker1_11.wav"

# Extract features
features = extract_features(file_path)
features = np.array(features).reshape(1, -1)

# Predict
prediction = model.predict(features)

if prediction[0] == 0:
    print("Speaker: speaker1")
else:
    print("Speaker: speaker2")