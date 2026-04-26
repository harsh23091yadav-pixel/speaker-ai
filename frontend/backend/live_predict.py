import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import joblib
from feature import extract_features

# Load model
model = joblib.load("models/speaker_model.pkl")

# Record settings
fs = 44100
seconds = 5
temp_file = "temp.wav"

print("🎤 Speak now...")

recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
sd.wait()

write(temp_file, fs, recording)

# Extract features
features = extract_features(temp_file)
features = np.array(features).reshape(1, -1)

# Predict
prediction = model.predict(features)

if prediction[0] == 0:
    print("Speaker: speaker1")
else:
    print("Speaker: speaker2")
