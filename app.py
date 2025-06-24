"""app.py
Streamlit Emotion Classification App

Run with:
    streamlit run app.py
"""

import streamlit as st
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Apply custom styling
st.set_page_config(
    page_title="Speech Emotion Classifier",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #1E3A8A;
        margin-top: 1rem;
    }
    .result-box {
        background-color: #F3F4F6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .footer {
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #E5E7EB;
        color: #6B7280;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

# Try both model formats
KERAS_MODEL_PATH = Path(__file__).parent / "emotion_classifier_model.keras"
H5_MODEL_PATH = Path(__file__).parent / "emotion_classifier_model.h5"
EMOTIONS = ['angry', 'calm', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
COLORS = ['#EF4444', '#10B981', '#7C3AED', '#3B82F6', '#F59E0B', '#6B7280', '#6B7280', '#EC4899']

@st.cache_resource(show_spinner=False)
def load_model_and_scaler():
    """Load model and create scaler from training data statistics"""
    
    # Load model
    model = None
    try:
        if KERAS_MODEL_PATH.exists():
            st.sidebar.write(f"Loading .keras model from {KERAS_MODEL_PATH}")
            model = tf.keras.models.load_model(KERAS_MODEL_PATH, compile=False)
            st.sidebar.write("✅ Model loaded successfully (.keras format)")
        elif H5_MODEL_PATH.exists():
            st.sidebar.write(f"Loading .h5 model from {H5_MODEL_PATH}")
            model = tf.keras.models.load_model(H5_MODEL_PATH, compile=False)
            st.sidebar.write("✅ Model loaded successfully (.h5 format)")
        else:
            st.error(f"❌ No model found at {KERAS_MODEL_PATH} or {H5_MODEL_PATH}")
            return None, None
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None
    
    # Create scaler from training data
    try:
        st.sidebar.write("Loading training data to fit scaler...")
        features_df = pd.read_csv("features.csv")
        
        # Extract feature columns (0-161)
        feature_cols = [str(i) for i in range(162)]
        X_train = features_df[feature_cols].values
        
        # Fit scaler on training data
        scaler = StandardScaler()
        scaler.fit(X_train)
        
        st.sidebar.write("✅ Scaler fitted on training data")
        st.sidebar.write(f"Training data shape: {X_train.shape}")
        
        return model, scaler
        
    except Exception as e:
        st.error(f"❌ Error creating scaler: {str(e)}")
        return model, None

def extract_features(y: np.ndarray, sr: int) -> np.ndarray:
    """Extract the 162‑dimension feature vector expected by the model."""
    result = np.array([])
    # 1. Zero‑crossing rate (1 dim)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y).T, axis=0)
    result = np.hstack((result, zcr))

    # 2. Chroma STFT (12 dim)
    stft = np.abs(librosa.stft(y))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
    result = np.hstack((result, chroma_stft))

    # 3. MFCC (20 dim)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20).T, axis=0)
    result = np.hstack((result, mfcc))

    # 4. Root Mean Square (1 dim)
    rms = np.mean(librosa.feature.rms(y=y).T, axis=0)
    result = np.hstack((result, rms))

    # 5. Mel‑spectrogram (128 dim)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
    result = np.hstack((result, mel))

    return result


def predict_emotion(wav_path: Path):
    """Compute prediction and return (label, probabilities)."""
    model, scaler = load_model_and_scaler()
    
    if model is None:
        st.error("❌ Failed to load model. Cannot make predictions.")
        return "unknown", [0] * len(EMOTIONS)
        
    if scaler is None:
        st.error("❌ Failed to create scaler. Cannot make predictions.")
        return "unknown", [0] * len(EMOTIONS)
        
    # Load and process audio
    y, sr = librosa.load(wav_path, duration=2.5, offset=0.6)  # 2.5s clip, skip 0.6s of leading silence
    
    # Debug: Check audio properties
    st.sidebar.write("🔍 Debug Info:")
    st.sidebar.write(f"Audio duration: {librosa.get_duration(y=y, sr=sr):.2f}s")
    st.sidebar.write(f"Sample rate: {sr} Hz")
    st.sidebar.write(f"Max amplitude: {np.max(np.abs(y)):.4f}")
    
    # Extract features
    features = extract_features(y, sr)
    
    # Debug: Check feature vector before scaling
    st.sidebar.write(f"Raw features shape: {features.shape}")
    st.sidebar.write(f"Raw features range: [{np.min(features):.4f}, {np.max(features):.4f}]")
    
    # 🚨 CRITICAL FIX: Apply StandardScaler to features before prediction
    features_scaled = scaler.transform(features.reshape(1, -1))[0]
    
    # Debug: Check scaled features
    st.sidebar.write(f"Scaled features range: [{np.min(features_scaled):.4f}, {np.max(features_scaled):.4f}]")
    
    # Reshape to (1, 162, 1) as used in training (Conv1D expecting (time_steps, 1))
    features_reshaped = np.expand_dims(features_scaled, axis=(0, 2))
    probs = model.predict(features_reshaped, verbose=0)[0]
    
    # Debug: Show raw probabilities
    st.sidebar.write("🎯 Prediction probabilities:")
    for emotion, prob in zip(EMOTIONS, probs):
        st.sidebar.write(f"  {emotion}: {prob:.4f}")
    
    idx = int(np.argmax(probs))
    
    # Debug: Check if probabilities are very close
    sorted_probs = sorted(probs, reverse=True)
    if len(sorted_probs) > 1 and sorted_probs[0] - sorted_probs[1] < 0.1:
        st.sidebar.write("⚠️ Close prediction! Top emotions are very similar.")
    
    return EMOTIONS[idx], probs


def main():
    st.markdown("<h1 class='main-header'>Speech Emotion Classifier</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write("Upload an audio file to analyze the emotional tone of speech.")
        
        uploaded_file = st.file_uploader("Choose a .wav file", type=["wav", "wave"])
    
    with col2:
        st.markdown("""
        <div style="background-color: #EFF6FF; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #3B82F6;">
            <p><strong>Supported emotions:</strong></p>
            <ul>
                <li>Angry</li>
                <li>Calm</li>
                <li>Disgust</li>
                <li>Fear</li>
                <li>Happy</li>
                <li>Neutral</li>
                <li>Sad</li>
                <li>Surprise</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    if uploaded_file is not None:
        wav_temp = Path("temp_audio.wav")
        wav_temp.write_bytes(uploaded_file.read())

        with st.spinner("Analyzing audio..."):
            label, probs = predict_emotion(wav_temp)
        
        st.markdown("<div class='result-box'>", unsafe_allow_html=True)
        st.markdown(f"<h2 class='sub-header'>Analysis Results</h2>", unsafe_allow_html=True)
        
        col_res1, col_res2 = st.columns([1, 1])
        
        with col_res1:
            st.markdown(f"<h3>Detected emotion: <span style='color: #1E3A8A;'>{label.capitalize()}</span></h3>", unsafe_allow_html=True)
            
            # Create probability bar chart with custom colors
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(EMOTIONS, probs, color=COLORS)
            ax.set_ylabel("Probability", fontsize=12)
            ax.set_ylim(0, 1)
            ax.set_xticklabels(EMOTIONS, rotation=45, ha='right', fontsize=10)
            ax.set_title("Emotion Probability Distribution", fontsize=14)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Highlight the predicted emotion
            bars[EMOTIONS.index(label)].set_edgecolor('black')
            bars[EMOTIONS.index(label)].set_linewidth(2)
            
            st.pyplot(fig)
        
        with col_res2:
            # Audio playback
            st.audio(uploaded_file, format="audio/wav")
            
            # Show waveform
            y, sr = librosa.load(wav_temp, duration=2.5, offset=0.6)
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            librosa.display.waveshow(y, sr=sr, ax=ax2, color='#3B82F6')
            ax2.set_title("Audio Waveform", fontsize=14)
            ax2.set_xlabel("Time (s)", fontsize=12)
            ax2.set_ylabel("Amplitude", fontsize=12)
            st.pyplot(fig2)
            
            # Add spectrogram for more visual information
            fig3, ax3 = plt.subplots(figsize=(10, 4))
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
            librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax3, cmap='viridis')
            ax3.set_title("Audio Spectrogram", fontsize=14)
            fig3.colorbar(ax3.collections[0], ax=ax3, format="%+2.f dB")
            st.pyplot(fig3)

        st.markdown("</div>", unsafe_allow_html=True)
        
        # Clean up
        wav_temp.unlink(missing_ok=True)

    st.markdown("<div class='footer'>Model: Conv1D network trained on the RAVDESS dataset (8 emotion classes)</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()