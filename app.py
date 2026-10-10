import streamlit as st
import torch
import torch.nn as nn
import librosa
import numpy as np
import tempfile
import os

from transformers import Wav2Vec2Model


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Speech Emotion Detection",
    page_icon="🎙️",
    layout="centered"
)


# ============================================================
# TITLE
# ============================================================

st.title("🎙️ Speech Emotion Detection")
st.write(
    "Upload an audio file and detect the emotion expressed in the speech."
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# EMOTION LABELS
# IMPORTANT: These must match the training label mapping.
# ============================================================

emotion_labels = {
    0: "neutral",
    1: "calm",
    2: "happy",
    3: "sad",
    4: "angry",
    5: "fearful",
    6: "disgust",
    7: "surprised"
}


NUM_CLASSES = 8


# ============================================================
# MODEL ARCHITECTURE
# Same architecture used in Colab
# ============================================================

class EmotionClassifier(nn.Module):

    def __init__(
        self,
        num_classes=NUM_CLASSES,
        freeze_backbone=True
    ):

        super(EmotionClassifier, self).__init__()

        self.backbone = Wav2Vec2Model.from_pretrained(
            "superb/wav2vec2-base-superb-er"
        )

        if freeze_backbone:

            for param in self.backbone.parameters():
                param.requires_grad = False

        hidden_size = self.backbone.config.hidden_size

        self.classifier = nn.Sequential(

            nn.Linear(
                hidden_size * 2,
                512
            ),

            nn.ReLU(),

            nn.BatchNorm1d(512),

            nn.Dropout(0.3),


            nn.Linear(
                512,
                256
            ),

            nn.ReLU(),

            nn.BatchNorm1d(256),

            nn.Dropout(0.3),


            nn.Linear(
                256,
                128
            ),

            nn.ReLU(),

            nn.BatchNorm1d(128),

            nn.Dropout(0.2),


            nn.Linear(
                128,
                num_classes
            )
        )


    def forward(self, audio):

        with torch.no_grad():

            outputs = self.backbone(audio)

            hidden_states = outputs.last_hidden_state


        # Mean pooling
        mean_pool = torch.mean(
            hidden_states,
            dim=1
        )


        # Max pooling
        max_pool = torch.max(
            hidden_states,
            dim=1
        )[0]


        # Combine mean + max
        pooled = torch.cat(
            [
                mean_pool,
                max_pool
            ],
            dim=1
        )


        # Classification
        logits = self.classifier(
            pooled
        )

        return logits


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = EmotionClassifier(
        num_classes=NUM_CLASSES,
        freeze_backbone=True
    )


    checkpoint = torch.load(
        "final_model_ravdess_sp_emotion.pth",
        map_location=device
    )


    model.load_state_dict(
        checkpoint
    )


    model.to(device)

    model.eval()

    return model


model = load_model()


# ============================================================
# AUDIO PREPROCESSING + PREDICTION
# ============================================================

def predict_emotion(audio_path):

    # Load audio
    waveform, sr = librosa.load(
        audio_path,
        sr=16000,
        mono=True
    )


    # 4 seconds
    max_length = 16000 * 4


    # Pad short audio
    if len(waveform) < max_length:

        waveform = np.pad(
            waveform,
            (
                0,
                max_length - len(waveform)
            )
        )


    # Cut long audio
    else:

        waveform = waveform[
            :max_length
        ]


    # Convert to tensor
    audio_tensor = torch.tensor(
        waveform,
        dtype=torch.float32
    ).unsqueeze(0)


    audio_tensor = audio_tensor.to(
        device
    )


    # Prediction
    with torch.no_grad():

        outputs = model(
            audio_tensor
        )


        probabilities = torch.softmax(
            outputs,
            dim=1
        )


    # Highest probability
    confidence, predicted_class = torch.max(
        probabilities,
        dim=1
    )


    emotion = emotion_labels[
        predicted_class.item()
    ]


    confidence = (
        confidence.item() * 100
    )


    return emotion, confidence, probabilities


# ============================================================
# AUDIO FILE UPLOAD
# ============================================================

st.subheader("📁 Upload Audio")


uploaded_file = st.file_uploader(
    "Choose an audio file",
    type=[
        "wav",
        "mp3",
        "ogg",
        "flac",
        "m4a"
    ]
)


if uploaded_file is not None:

    # Play uploaded audio
    st.audio(
        uploaded_file
    )


    # Detect button
    if st.button(
        "🔍 Detect Emotion"
    ):

        # Get file extension
        suffix = os.path.splitext(
            uploaded_file.name
        )[1]


        # Create temporary file
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_audio:

            temp_audio.write(
                uploaded_file.read()
            )

            temp_audio_path = (
                temp_audio.name
            )


        try:

            # Predict
            emotion, confidence, probabilities = (
                predict_emotion(
                    temp_audio_path
                )
            )


            # Display result
            st.success(
                f"🎭 Detected Emotion: {emotion.capitalize()}"
            )


            st.info(
                f"📊 Confidence: {confidence:.2f}%"
            )


            # Display probability for all emotions
            st.subheader(
                "Emotion Probabilities"
            )


            probability_values = (
                probabilities[0]
                .cpu()
                .numpy()
                * 100
            )


            for index, value in enumerate(
                probability_values
            ):

                st.write(
                    f"{emotion_labels[index].capitalize()}: "
                    f"{value:.2f}%"
                )


                st.progress(
                    float(value / 100)
                )


        except Exception as e:

            st.error(
                f"Error while processing audio: {e}"
            )


        finally:

            # Delete temporary file
            if os.path.exists(
                temp_audio_path
            ):

                os.remove(
                    temp_audio_path
                )