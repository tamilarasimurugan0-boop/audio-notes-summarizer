import os
import tempfile

import streamlit as st
from faster_whisper import WhisperModel
from transformers import pipeline


st.title("Meeting / Lecture Notes Summarizer")
st.write("Upload an audio recording to get its transcript and summary.")

# Model once load aagi app-la reuse aagum
@st.cache_resource
def load_models():
    speech_model = WhisperModel("base", device="cpu", compute_type="int8")
    summary_model = pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6"
    )
    return speech_model, summary_model


audio = st.file_uploader(
    "Choose an audio file",
    type=["mp3", "wav", "m4a"]
)

if audio and st.button("Generate Notes"):
    audio_path = None

    try:
        speech_model, summary_model = load_models()

        # Uploaded audio-va temporary file-aa save pannrom
        suffix = os.path.splitext(audio.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(audio.getvalue())
            audio_path = f.name

        with st.spinner("Transcribing audio..."):
            segments, info = speech_model.transcribe(
                audio_path,
                vad_filter=True
            )
            transcript = " ".join(
                segment.text.strip() for segment in segments
            ).strip()

        if not transcript:
            st.error("Audio-la speech detect aagala. Vera audio try pannunga.")
        else:
            # Long transcript-ai small chunks-aa pirikkrom
            words = transcript.split()
            chunks = [
                " ".join(words[i:i + 250])
                for i in range(0, len(words), 250)
            ]

            with st.spinner("Creating notes..."):
                summaries = []
                for chunk in chunks:
                    result = summary_model(
                        chunk,
                        max_length=100,
                        min_length=20,
                        do_sample=False
                    )
                    summaries.append(result[0]["summary_text"])

            st.subheader("Summary")
            st.write(" ".join(summaries))

            st.subheader("Transcript")
            st.text_area("Full transcript", transcript, height=250)

            notes = "SUMMARY\n" + " ".join(summaries)
            notes += "\n\nFULL TRANSCRIPT\n" + transcript

            st.download_button(
                "Download notes",
                data=notes,
                file_name="meeting_notes.txt",
                mime="text/plain"
            )

    except Exception as error:
        st.error(f"Something went wrong: {error}")

    finally:
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)