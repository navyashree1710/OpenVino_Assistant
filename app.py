from flask import Flask, render_template, request, jsonify
import speech_recognition as sr
import os
import cv2
from PIL import Image
from transformers import pipeline
import numpy as np
from fer import FER
import base64
import io

app = Flask(__name__)
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

def speak(text):
    escaped_text = text.replace("'", " ")
    os.system(f'powershell -c "Add-Type –AssemblyName System.Speech; $speak = New-Object System.Speech.Synthesis.SpeechSynthesizer; $speak.Speak(\'{escaped_text}\')"')

def listen():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        audio = recognizer.listen(source)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "Sorry, I couldn't understand you."

def analyze_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    return f"Processed image shape: {gray.shape}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    question = request.json['question']
    context = """
        Artificial Intelligence (AI) is the simulation of human intelligence by machines.
        It includes learning, reasoning, problem-solving, and understanding language.
        Machine learning is a subfield of AI that enables computers to learn from data.
    """
    answer = qa_pipeline({'context': context, 'question': question})['answer']
    return jsonify({"answer": answer})

@app.route('/voice', methods=['POST'])
def voice():
    query = listen()
    context = """
        Artificial Intelligence (AI) is the simulation of human intelligence by machines.
        It includes learning, reasoning, problem-solving, and understanding language.
        Machine learning is a subfield of AI that enables computers to learn from data.
    """
    answer = qa_pipeline({'context': context, 'question': query})['answer']
    return jsonify({"transcript": query, "answer": answer})

@app.route('/upload-image', methods=['POST'])
def upload_image():
    image_data = request.files['image'].read()
    result = analyze_image(image_data)
    return jsonify({"result": result})

@app.route('/emotion', methods=['POST'])
def emotion():
    file = request.files['frame']
    frame_bytes = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(frame_bytes, cv2.IMREAD_COLOR)
    detector = FER()
    result = detector.top_emotion(frame)
    if result and result[1] is not None:
        emotion = result[0]
        confidence = int(result[1] * 100)
        message = "Well looks like you understood the concept perfectly." if emotion == 'happy' else "Did you not understand? Want me to explain again?"
        return jsonify({"emotion": emotion, "confidence": confidence, "message": message})
    return jsonify({"emotion": "none", "confidence": 0, "message": "No clear emotion detected."})

if __name__ == '__main__':
    app.run(debug=True)




