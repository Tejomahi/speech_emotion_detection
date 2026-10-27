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

        st.code("final_model_ravdess_sp_emotion.pth")

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
