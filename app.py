import os
import traceback

import speech_recognition as sr
import streamlit as st
from transformers import pipeline

# Set Streamlit app layout to wide
st.set_page_config(layout="wide")

# Designing the interface
st.title("🎧 Audio Analysis 📝")

# Define Streamlit app sidebar
st.sidebar.title("Audio Analysis")
st.sidebar.write("The Audio Analysis app is a powerful tool that allows you to analyze audio files and gain valuable insights from them."
                 "It combines speech recognition and sentiment analysis techniques to transcribe the audio and determine the sentiment expressed within it.")

# Upload audio file
st.sidebar.header("Upload Audio")
audio_file = st.sidebar.file_uploader("Browse", type=["wav"])
upload_button = st.sidebar.button("Upload")


def perform_sentiment_analysis(text):
    # Load the sentiment analysis model
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    sentiment_analysis = pipeline("sentiment-analysis", model=model_name)

    # Perform sentiment analysis on the text
    results = sentiment_analysis(text)

    # Extract the sentiment label and score
    sentiment_label = results[0]['label']
    sentiment_score = results[0]['score']

    return sentiment_label, sentiment_score


def transcribe_audio(audio_file):
    r = sr.Recognizer()

    with sr.AudioFile(audio_file) as source:
        audio = r.record(source)  # Read the entire audio file

    transcribed_text = r.recognize_google(audio)  # Perform speech recognition

    return transcribed_text


def main():
    # Perform analysis when audio file is uploaded
    if audio_file and upload_button:
        try:
            # Perform audio transcription
            transcribed_text = transcribe_audio(audio_file)

            # Perform sentiment analysis
            sentiment_label, sentiment_score = perform_sentiment_analysis(
                transcribed_text)

            # Display the results
            st.header("Transcribed Text")
            st.text_area("Transcribed Text", transcribed_text, height=200)

            st.header("Sentiment Analysis")

            # Display sentiment labels with icons and scores
            negative_icon = "👎"
            neutral_icon = "😐"
            positive_icon = "👍"

            if sentiment_label == "NEGATIVE":
                st.write(
                    f"{negative_icon} Negative (Score: {sentiment_score})", unsafe_allow_html=True)
            else:
                st.empty()

            if sentiment_label == "NEUTRAL":
                st.write(
                    f"{neutral_icon} Neutral (Score: {sentiment_score})", unsafe_allow_html=True)
            else:
                st.empty()

            if sentiment_label == "POSITIVE":
                st.write(
                    f"{positive_icon} Positive (Score: {sentiment_score})", unsafe_allow_html=True)
            else:
                st.empty()

            # Provide additional information about sentiment score interpretation
            st.info("The sentiment score assesses the intensity of positive, negative, or neutral emotions or opinions "
                    "A higher score indicates a stronger sentiment, while a lower score indicates a weaker sentiment.")

        except Exception as ex:
            st.error(
                "Error occurred during audio transcription and sentiment analysis.")
            st.error(str(ex))
            traceback.print_exc()


if __name__ == "__main__":
    main()
