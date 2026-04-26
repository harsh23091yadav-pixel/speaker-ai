import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib

from train import X, y

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = SVC(kernel='linear', probability=True)

# Train
model.fit(X_train, y_train)

# Accuracy
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)

# Save model
joblib.dump(model, "models/speaker_model.pkl")
print("Model saved!")
