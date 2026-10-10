
import os
import tempfile
from io import BytesIO

import librosa
import numpy as np
import requests
import streamlit as st
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model


# ============================================
# 1. APPLICATION SETTINGS
# ============================================

st.set_page_config(
    page_title="Speech Emotion Detection",
    page_icon="🎙️",
    layout="centered"
)

st.title("🎙️ Speech Emotion Detection")
st.write(
    "Detect emotions in speech using a deep learning model."
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

NUM_CLASSES = 8

emotion_labels = {
    0: "Neutral",
    1: "Calm",
    2: "Happy",
    3: "Sad",
    4: "Angry",
    5: "Fearful",
    6: "Disgust",
    7: "Surprised"
}

MODEL_URL = (
    "https://huggingface.co/Tejomahi/"
    "speech_emotion_detection_model/resolve/main/"
    "final_model_ravdess_sp_emotion.pth"
)


# ============================================
# 2. MODEL ARCHITECTURE
# Must match the training notebook
# ============================================

class EmotionClassifier(nn.Module):

    def __init__(
        self,
        num_classes=NUM_CLASSES,
        freeze_backbone=True
    ):
        super().__init__()

        self.backbone = Wav2Vec2Model.from_pretrained(
            "superb/wav2vec2-base-superb-er"
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        hidden_size = self.backbone.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),

            nn.Linear(128, num_classes)
        )

    def forward(self, audio):

        with torch.no_grad():
            outputs = self.backbone(audio)
            hidden_states = outputs.last_hidden_state

        mean_pool = torch.mean(hidden_states, dim=1)
        max_pool = torch.max(hidden_states, dim=1)[0]

        pooled = torch.cat(
            [mean_pool, max_pool],
            dim=1
        )

        return self.classifier(pooled)


# ============================================
# 3. LOAD MODEL FROM HUGGING FACE
# ============================================

@st.cache_resource
def load_model():

    model = EmotionClassifier(
        num_classes=NUM_CLASSES,
        freeze_backbone=True
    )

    response = requests.get(
        MODEL_URL,
        timeout=300
    )
    response.raise_for_status()

    checkpoint = torch.load(
        BytesIO(response.content),
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    return model


# ============================================
# 4. AUDIO PREDICTION
# ============================================

def predict_emotion(audio_path):

    waveform, sr = librosa.load(
        audio_path,
        sr=16000,
        mono=True
    )

    if len(waveform) == 0:
        raise ValueError("The audio file is empty.")

    # Use exactly 4 seconds, as in the inference setup
    max_length = 16000 * 4

    if len(waveform) < max_length:
        waveform = np.pad(
            waveform,
            (0, max_length - len(waveform))
        )
    else:
        waveform = waveform[:max_length]

    audio_tensor = torch.tensor(
        waveform,
        dtype=torch.float32
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(audio_tensor)
        probabilities = torch.softmax(outputs, dim=1)

    confidence, predicted_class = torch.max(
        probabilities,
        dim=1
    )

    emotion = emotion_labels[predicted_class.item()]
    confidence_percent = confidence.item() * 100

    probability_values = (
        probabilities[0].cpu().numpy() * 100
    )

    return emotion, confidence_percent, probability_values


# ============================================
# 5. DISPLAY PREDICTION
# ============================================

def show_prediction(audio_bytes, suffix):

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_file:
            temp_file.write(audio_bytes)
            temp_path = temp_file.name

        with st.spinner("Analyzing your audio..."):

            emotion, confidence, probabilities = (
                predict_emotion(temp_path)
            )

        st.success(f"🎭 Detected Emotion: {emotion}")
        st.metric(
            "Model Confidence",
            f"{confidence:.2f}%"
        )

        st.subheader("Emotion Probabilities")

        for index, value in enumerate(probabilities):
            st.write(
                f"{emotion_labels[index]}: {value:.2f}%"
            )
            st.progress(
                float(value / 100)
            )

        st.caption(
            "Predictions are estimates from the trained model, "
            "not definitive measurements of a person's feelings."
        )

    except Exception as error:
        st.error(
            f"Could not process the audio: {error}"
        )

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# ============================================
# 6. LOAD MODEL
# ============================================

try:
    with st.spinner(
        "Loading the emotion model. "
        "The first launch may take several minutes..."
    ):
        model = load_model()

    st.success("Model loaded successfully.")

except Exception as error:
    st.error("The model could not be loaded.")
    st.code(str(error))
    st.info(
        "Check the Hugging Face model filename, repository "
        "access, network connection, and deployment logs."
    )
    st.stop()


# ============================================
# 7. INPUT METHOD A: AUDIO FILE UPLOAD
# ============================================

st.divider()
st.subheader("📁 Upload an Audio File")

uploaded_file = st.file_uploader(
    "Choose an audio recording",
    type=["wav", "mp3", "ogg", "flac", "m4a"]
)

if uploaded_file is not None:

    st.audio(uploaded_file)

    if st.button(
        "🔍 Detect Emotion from File",
        key="file_predict"
    ):
        suffix = os.path.splitext(
            uploaded_file.name
        )[1].lower()

        show_prediction(
            uploaded_file.getvalue(),
            suffix or ".wav"
        )


# ============================================
# 8. INPUT METHOD B: MICROPHONE RECORDING
# ============================================

st.divider()
st.subheader("🎙️ Record Using Your Microphone")

st.write(
    "Allow microphone access in your browser, "
    "record your speech, and submit the recording."
)

recorded_audio = st.audio_input(
    "Record your voice",
    sample_rate=16000
)

if recorded_audio is not None:

    st.audio(recorded_audio)

    if st.button(
        "🔍 Detect Emotion from Recording",
        key="mic_predict"
    ):
        show_prediction(
            recorded_audio.getvalue(),
            ".wav"
        )


# ============================================
# 9. FOOTER
# ============================================

st.divider()

st.caption(
    "SpeechSense | Deep Learning-Based Speech "
    "Emotion Recognition | 8 emotion classes"
)
