
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
    layout="centered",
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
    "final_model_ravdess_sp_emotion.pth"
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
    "surprised",
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

        self.backbone = Wav2Vec2Model.from_pretrained(MODEL_NAME)

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
            nn.Linear(128, num_classes),
        )

    def forward(self, audio):
        outputs = self.backbone(audio)
        hidden_states = outputs.last_hidden_state
        mean_pool = torch.mean(hidden_states, dim=1)
        max_pool = torch.max(hidden_states, dim=1)[0]
        pooled = torch.cat([mean_pool, max_pool], dim=1)
        return self.classifier(pooled)


# --------------------------------------------------
# 4. DOWNLOAD AND LOAD THE TRAINED MODEL
# --------------------------------------------------
