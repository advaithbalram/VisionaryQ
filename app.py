from flask import Flask, request, jsonify, render_template
from nltk.tokenize import word_tokenize
from pydub import AudioSegment
from sentence_transformers import SentenceTransformer
import moviepy.editor as mp
import speech_recognition as sr
import re, nltk

# Download tokenizer
nltk.download("punkt")

# Initialize Flask app
app = Flask(__name__)

def video_to_text(video_path, audio_output="video.wav"):
    try:
        # Load the video
        video = mp.VideoFileClip(video_path)

        # Extract and save the audio from the video
        audio_file = video.audio
        audio_file.write_audiofile(audio_output)

        # Initialize the recognizer
        recognizer = sr.Recognizer()

        # Load and transcribe the audio file
        with sr.AudioFile(audio_output) as source:
            audio_data = recognizer.record(source)

        # Convert speech to text
        text = recognizer.recognize_google(audio_data)

        return text

    except Exception as e:
        return f"Error: {str(e)}"
    
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'\[.*?\]', '', text)  # Remove non-verbal sounds [laughter], [music]
    text = re.sub(r'[^a-zA-Z0-9,.!?\'"]+', ' ', text)  # Remove special characters
    return text.strip()

def split_text(text, max_words=100):
    words = word_tokenize(text)  # Tokenize text into words
    chunks = []
    chunk = []

    for word in words:
        if len(chunk) < max_words:
            chunk.append(word)
        else:
            chunks.append(" ".join(chunk))  # Join words into a chunk
            chunk = [word]  # Start a new chunk

    if chunk:  # Add any remaining words
        chunks.append(" ".join(chunk))

    return chunks

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
