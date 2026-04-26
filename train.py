import os
import numpy as np
from feature import extract_features

DATA_PATH = "data"

X = []
y = []

labels = {
    "speaker1": 0,
    "speaker2": 1
}

for speaker in labels:
    folder = os.path.join(DATA_PATH, speaker)

    for file in os.listdir(folder):
        if file.endswith(".wav"):
            file_path = os.path.join(folder, file)

            features = extract_features(file_path)
            X.append(features)
            y.append(labels[speaker])

X = np.array(X)
y = np.array(y)

print("X shape:", X.shape)
print("y:", y)