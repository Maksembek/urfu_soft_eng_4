import whisper
import streamlit as st

st.title("Whisper App")

# upload audio file with streamlit
audio_file = st.file_uploader("Upload Audio", type=["wav", "mp3", "m4a"])

model = whisper.load_model("base")
st.text("Whisper Model Loaded")

st.text("Play Original Audio File")
st.audio(audio_file)

if st.button("Transcribe Audio"):
    if audio_file is None:
        st.error("Please upload an audio file")
        exit(1)

    st.success("Transcribing Audio")
    transcription = model.transcribe(audio_file.name, fp16=False, language='ru')
    st.success("Transcription Complete")
    st.markdown(transcription["text"])
