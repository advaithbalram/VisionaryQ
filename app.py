from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from nltk.tokenize import word_tokenize
from pydub import AudioSegment
from sentence_transformers import SentenceTransformer
from llama_index.llms.groq import Groq
from llama_index.core.llms import ChatMessage, MessageRole
from pinecone import Pinecone, ServerlessSpec
import moviepy.editor as mp
import speech_recognition as sr
import re, nltk, os

load_dotenv()

# Download tokenizer
nltk.download("punkt")

# Initialize Flask app
app = Flask(__name__)

# Initialize embedding model (384 dimensions for all-MiniLM-L6-v2)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize Pinecone
INDEX_NAME = "video-transcriptions"
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Check if index exists, create only if it does not
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if INDEX_NAME not in existing_indexes:
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="dotproduct",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print(f"Index '{INDEX_NAME}' created.")
else:
    print(f"Index '{INDEX_NAME}' already exists. Skipping creation.")

# Connect to the existing Pinecone index
index = pc.Index(INDEX_NAME)

# Function to extract text from video
def video_to_text(video_path, chunk_length=30):
    try:
        video = mp.VideoFileClip(video_path)
        audio_path = "video_audio.wav"
        video.audio.write_audiofile(audio_path)
        video.close()

        audio = AudioSegment.from_file(audio_path)
        audio = audio.set_channels(1)  # Convert to mono
        audio.export(audio_path, format="wav")

        duration = len(audio) / 1000  # Convert milliseconds to seconds
        print(f"Total Audio Duration: {duration} seconds")

        recognizer = sr.Recognizer()
        transcribed_text = []

        for i in range(0, int(duration), chunk_length):
            with sr.AudioFile(audio_path) as source:
                audio_data = recognizer.record(source, offset=i, duration=chunk_length)
                try:
                    chunk_text = recognizer.recognize_google(audio_data)
                    transcribed_text.append(chunk_text)
                except sr.UnknownValueError:
                    print(f"Chunk {i}-{i+chunk_length} could not be transcribed.")
                except sr.RequestError as e:
                    print(f"Google API request failed: {e}")
        os.remove(audio_path)  # Remove temp audio file after processing
        return " ".join(transcribed_text)

    except Exception as e:
        return f"Error: {str(e)}"

# Function to clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"[^a-zA-Z0-9,.!?\'\"]+", " ", text)
    return text.strip()

# Function to split text into chunks
def split_text(text, max_words=100):
    words = word_tokenize(text)
    chunks = []
    chunk = []

    for word in words:
        if len(chunk) < max_words:
            chunk.append(word)
        else:
            chunks.append(" ".join(chunk))
            chunk = [word]

    if chunk:
        chunks.append(" ".join(chunk))

    return chunks

def store_chunks_in_pinecone(video_name, text_chunks):
    # Delete all previous embeddings before storing new ones
    index.delete(delete_all=True)

    upsert_data = []
    for i, chunk in enumerate(text_chunks):
        chunk_id = f"{video_name}_{i}"
        vector = embedding_model.encode(chunk).tolist()
        metadata = {
            "text": chunk,
            "video_name": video_name,
            "chunk_id": chunk_id,
        }
        upsert_data.append((chunk_id, vector, metadata))

    if upsert_data:
        index.upsert(upsert_data)

# Function to retrieve relevant chunks from Pinecone
def retrieve_relevant_chunks(query, top_k=3):
    query_vector = embedding_model.encode(query).tolist()
    results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
    return [match["metadata"]["text"] for match in results.get("matches", [])]

# Function to generate answers using Groq API
def get_answer(query):
    relevant_chunks = retrieve_relevant_chunks(query)
    context = " ".join(relevant_chunks)

    llm = Groq(api_key=os.getenv("GROQ_API_KEY"), model="llama3-8b-8192")
    system_message = (
        "You are VisionaryQ, an AI assistant that answers questions strictly based on the provided video."
        "If the answer is not found in the video, respond with: 'I'm sorry, but I cannot find that information in the video provided.'"
    )
    general_greetings = ["hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening"]
    if query.lower() in general_greetings:
        return "Hello! I'm VisionaryQ. I can answer questions based on the video provided. What would you like to ask?"
    
    prompt = f"Use the following context to answer the query: {context}\n\nQuery: {query}"

    # Ensure the messages are formatted properly
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content=system_message),
        ChatMessage(role=MessageRole.USER, content=prompt)
    ]

    response = llm.chat(messages)  # Ensure messages are in the correct format
    if not relevant_chunks:
        return "I'm sorry, but I cannot find that information in the video provided."
    
    return response.message.content  # Extract the response text

@app.route("/")
def home():
    return render_template("index.html")

# Flask API - Upload and Process Video
@app.route("/upload", methods=["POST"])
def upload_video():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    temp_path = os.path.join(os.getcwd(), file.filename)  # Use a safe temp path
    file.save(temp_path)

    try:
        video_name = file.filename
        transcribed_text = video_to_text(temp_path)
    finally:
        os.remove(temp_path)  # Delete file after processing

    cleaned_text = clean_text(transcribed_text)
    text_chunks = split_text(cleaned_text)
    store_chunks_in_pinecone(video_name, text_chunks)

    return jsonify({"message": "Video processed and stored"}), 200

# Flask API - Ask Questions
@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.json
    query = data.get("query", "")

    if not query:
        return jsonify({"error": "No query provided"}), 400

    answer = get_answer(query)
    return jsonify({"answer": answer})

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
