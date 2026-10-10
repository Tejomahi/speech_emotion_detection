
import io
import requests
import streamlit as st
import numpy as np
import librosa
import torch
import torch.nn as nn

from transformers import Wav2Vec2Model


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎙️",
    layout="centered"
)

st.title("🎙️ Speech Emotion Recognition")
st.write(
    "Upload an audio file or record your voice to identify "
    "the emotion expressed in the speech."
)


# --------------------------------------------------
# 2. MODEL CONFIGURATION
# --------------------------------------------------

MODEL_URL = (
    "https://huggingface.co/Tejomahi/"
    "speech_emotion_detection_model/resolve/main/"
    "final_speech_emotion_detection_model.pth"
)

MODEL_NAME = "superb/wav2vec2-base-superb-er"

EMOTIONS = [
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprised"
]

SAMPLE_RATE = 16000
AUDIO_DURATION = 4
NUM_SAMPLES = SAMPLE_RATE * AUDIO_DURATION

DEVICE = torch.device("cpu")


# --------------------------------------------------
# 3. MODEL ARCHITECTURE
# Must match the architecture used during training.
# --------------------------------------------------

class EmotionClassifier(nn.Module):

    def __init__(self, num_classes=8, freeze_backbone=True):
        super().__init__()

        self.backbone = Wav2Vec2Model.from_pretrained(
            MODEL_NAME
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

        outputs = self.backbone(audio)
        hidden_states = outputs.last_hidden_state

        mean_pool = torch.mean(hidden_states, dim=1)
        max_pool = torch.max(hidden_states, dim=1)[0]

        pooled = torch.cat(
            [mean_pool, max_pool],
            dim=1
        )

        logits = self.classifier(pooled)

        return logits


# --------------------------------------------------
# 4. DOWNLOAD AND LOAD THE TRAINED MODEL
# --------------------------------------------------

@st.cache_resource(show_spinner="Loading the trained AI model...")
def load_model():

    response = requests.get(
        MODEL_URL,
        timeout=300
    )

    response.raise_for_status()

    checkpoint = torch.load(
        io.BytesIO(response.content),
        map_location=DEVICE,
        weights_only=True
    )

    model = EmotionClassifier(
        num_classes=len(EMOTIONS),
        freeze_backbone=True
    )

    model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()

    return model


# --------------------------------------------------
# 5. PREPROCESS AUDIO
# Convert audio to mono, 16 kHz, and 4 seconds.
# --------------------------------------------------

def preprocess_audio(audio_file):

    audio_file.seek(0)

    audio_bytes = audio_file.read()

    audio, _ = librosa.load(
        io.BytesIO(audio_bytes),
        sr=SAMPLE_RATE,
        mono=True
    )

    if len(audio) == 0:
        raise ValueError("The audio file is empty.")

    # Pad short audio or truncate long audio.
    if len(audio) < NUM_SAMPLES:

        audio = np.pad(
            audio,
            (0, NUM_SAMPLES - len(audio)),
            mode="constant"
        )

    else:
        audio = audio[:NUM_SAMPLES]

    audio = np.asarray(audio, dtype=np.float32)

    return torch.tensor(
        audio,
        dtype=torch.float32
    ).unsqueeze(0)


# --------------------------------------------------
# 6. PREDICT EMOTION
# --------------------------------------------------

def predict_emotion(model, audio_file):

    input_audio = preprocess_audio(audio_file)
    input_audio = input_audio.to(DEVICE)

    with torch.inference_mode():

        logits = model(input_audio)

        probabilities = torch.softmax(
            logits,
            dim=1
        )[0]

        predicted_index = torch.argmax(
            probabilities
        ).item()

    predicted_emotion = EMOTIONS[predicted_index]
    confidence = probabilities[predicted_index].item() * 100

    probability_results = {
        EMOTIONS[i]: float(probabilities[i].item() * 100)
        for i in range(len(EMOTIONS))
    }

    return predicted_emotion, confidence, probability_results


# --------------------------------------------------
# 7. DISPLAY RESULTS
# --------------------------------------------------

def display_prediction(audio_file):

    try:
        with st.spinner("Analyzing your voice..."):

            model = load_model()

            emotion, confidence, probabilities = predict_emotion(
                model,
                audio_file
            )

        st.success("Emotion analysis completed!")

        st.subheader("Predicted Emotion")

        st.markdown(
            f"## 🎯 {emotion.capitalize()}"
        )

        st.metric(
            label="Model confidence",
            value=f"{confidence:.2f}%"
        )

        st.subheader("Emotion Probability Scores")

        sorted_probabilities = dict(
            sorted(
                probabilities.items(),
                key=lambda item: item[1],
                reverse=True
            )
        )

        for label, score in sorted_probabilities.items():

            st.write(
                f"**{label.capitalize()} — {score:.2f}%**"
            )

            st.progress(
                min(max(int(round(score)), 0), 100)
            )

        st.caption(
            "These percentages are model scores, not a definitive "
            "measurement of a person's actual emotional state."
        )

    except requests.exceptions.HTTPError as error:

        st.error(
            "The model file could not be downloaded from Hugging Face."
        )

        st.write(
            "Please check that the model file has finished uploading, "
            "is in the main branch, and has the exact filename:"
        )

        st.code("final_speech_emotion_detection_model.pth")

        st.caption(f"Details: {error}")

    except Exception as error:

        st.error(
            "Unable to analyze this audio. Please try another "
            "clear WAV, MP3, OGG, FLAC, or M4A recording."
        )

        st.caption(f"Technical details: {error}")


# --------------------------------------------------
# 8. AUDIO FILE UPLOAD
# --------------------------------------------------

st.divider()

st.subheader("📁 Option 1: Upload an Audio File")

uploaded_file = st.file_uploader(
    "Choose an audio file",
    type=["wav", "mp3", "ogg", "flac", "m4a"],
    key="uploaded_audio"
)

if uploaded_file is not None:

    st.audio(
        uploaded_file,
        format=uploaded_file.type
    )

    if st.button(
        "Analyze Uploaded Audio",
        type="primary",
        use_container_width=True
    ):

        display_prediction(uploaded_file)


# --------------------------------------------------
# 9. MICROPHONE RECORDING
# --------------------------------------------------

st.divider()

st.subheader("🎤 Option 2: Record Your Voice")

st.write(
    "Click the recording control below, record your voice, "
    "and finish the recording."
)

recorded_audio = st.audio_input(
    "Record your voice",
    key="recorded_audio"
)

if recorded_audio is not None:

    st.audio(recorded_audio)

    if st.button(
        "Analyze Recorded Voice",
        type="primary",
        use_container_width=True
    ):

        display_prediction(recorded_audio)


# --------------------------------------------------
# 10. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Speech Emotion Recognition using Deep Learning "
    "and Wav2Vec2 | Academic Project"
)
