import sounddevice as sd
from scipy.io.wavfile import write
import os

fs = 44100
seconds = 5

name = input("Enter speaker name (speaker1/speaker2): ")
file_num = input("Enter file number: ")

folder = f"data/{name}"
os.makedirs(folder, exist_ok=True)

filename = f"{folder}/{name}_{file_num}.wav"

print("Recording... Speak now")

recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
sd.wait()

write(filename, fs, recording)

print(f"Saved: {filename}")