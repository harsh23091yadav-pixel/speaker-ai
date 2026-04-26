import librosa
import numpy as np

def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, sr=None)

    mfccs = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=40
    )

    # Take mean across time
    mfccs_mean = np.mean(mfccs.T, axis=0)

    return mfccs_mean


if __name__ == "__main__":
    features = extract_features("output.wav")
    print("MFCC shape:", features.shape)
    print(features)
