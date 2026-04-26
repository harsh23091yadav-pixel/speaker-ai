import streamlit as st
import os
import numpy as np
import joblib
import tempfile
import time
import shutil

import librosa
import librosa.display
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

# ========= PATHS =========
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_PATH = os.path.join(BASE_DIR, "models", "speaker_model.pkl")

os.makedirs(DATA_DIR, exist_ok=True)

# ========= UI CONFIG =========
st.set_page_config(page_title="Speaker AI", layout="wide")

st.markdown("""
<style>
body, .stApp {
    background: linear-gradient(135deg, #0f172a, #020617);
    color: white;
}
.card {
    background: rgba(255,255,255,0.05);
    padding: 20px;
    border-radius: 16px;
    box-shadow: 0px 8px 20px rgba(0,0,0,0.4);
}
</style>
""", unsafe_allow_html=True)

st.title("🎤 Speaker Recognition (Auto AI)")

menu = st.sidebar.radio("Menu", ["Enroll", "Test", "Manage"])

# ========= FEATURE =========
def extract_features(file):
    y, sr = librosa.load(file, sr=None)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc

# ========= TRAIN =========
def train_model():
    X, y = [], []
    speakers = os.listdir(DATA_DIR)

    for i, spk in enumerate(speakers):
        spk_path = os.path.join(DATA_DIR, spk)

        for f in os.listdir(spk_path):
            try:
                path = os.path.join(spk_path, f)
                feat = extract_features(path)
                X.append(feat)
                y.append(i)
            except:
                pass

    if len(set(y)) < 2:
        return False

    model = RandomForestClassifier()
    model.fit(X, y)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    return True

# ========= ENROLL =========
if menu == "Enroll":

    col1, col2 = st.columns([2,1])

    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        st.subheader("🎙 Record Speaker")

        name = st.text_input("", placeholder="Enter speaker name")

        audio = st.audio_input("Record voice")

        if audio and name:
            spk_dir = os.path.join(DATA_DIR, name)
            os.makedirs(spk_dir, exist_ok=True)

            path = os.path.join(spk_dir, f"{int(time.time())}.wav")

            with open(path, "wb") as f:
                f.write(audio.read())

            st.success("Saved!")

            # auto train
            if train_model():
                st.info("Model trained")

        st.markdown("</div>", unsafe_allow_html=True)

    # ========= SAVED FILES =========
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("📁 Saved")

        speakers = os.listdir(DATA_DIR)

        if len(speakers) == 0:
            st.info("No data")
        else:
            for spk in speakers:

                spk_path = os.path.join(DATA_DIR, spk)

                c1, c2 = st.columns([3,1])

                with c1:
                    st.write(f"**{spk}**")

                with c2:
                    if st.button("🗑️", key=f"del_{spk}"):
                        shutil.rmtree(spk_path)
                        st.rerun()

                files = os.listdir(spk_path)

                for f in files:
                    f1, f2 = st.columns([3,1])

                    with f1:
                        st.write(f"🎧 {f}")

                    with f2:
                        if st.button("❌", key=f"{spk}_{f}"):
                            os.remove(os.path.join(spk_path, f))
                            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

# ========= TEST =========
elif menu == "Test":

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.subheader("🧠 Test Speaker")

    if not os.path.exists(MODEL_PATH):
        st.warning("Record at least 2 speakers first")
    else:
        model = joblib.load(MODEL_PATH)
        speakers = os.listdir(DATA_DIR)

        audio = st.audio_input("Speak now")

        if audio:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                f.write(audio.read())
                path = f.name

            y, sr = librosa.load(path)

            fig, ax = plt.subplots()
            librosa.display.waveshow(y, sr=sr, ax=ax)
            st.pyplot(fig)

            feat = extract_features(path)
            feat = np.array(feat).reshape(1, -1)

            pred = model.predict(feat)[0]
            prob = model.predict_proba(feat)[0]

            name = speakers[pred]
            conf = max(prob)

            st.success(f"🎯 {name}")
            st.progress(float(conf))

    st.markdown("</div>", unsafe_allow_html=True)

# ========= MANAGE =========
elif menu == "Manage":

    st.subheader("⚙️ Manage Data")

    if st.button("🧹 Clear All Data"):
        shutil.rmtree(DATA_DIR)
        os.makedirs(DATA_DIR, exist_ok=True)

        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)

        st.success("All data cleared")
