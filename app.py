import spacy
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import os
import random
import speech_recognition as sr
from flask import Flask, render_template, request
from logic.pronunciation import get_pronunciation
from logic.vocabulary import VocabularyInfoFetcher
from logic.grammar import correct_and_analyze
from gtts import gTTS

# Load custom grammar correction model
class GrammarCorrector:
    def __init__(self):
        self.model_name = "vennify/t5-base-grammar-correction"
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)

    def correct_grammar(self, text):
        input_text = "grammar: " + text
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt", truncation=True)

        with torch.no_grad():
            outputs = self.model.generate(input_ids, max_length=128, num_beams=4, early_stopping=True)

        corrected = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return corrected

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def correct_and_analyze(text):
    # Instantiate GrammarCorrector class
    grammar_corrector = GrammarCorrector()

    # Correct the sentence using the custom model
    corrected = grammar_corrector.correct_grammar(text)

    # Debugging output: print the corrected sentence
    print(f"Corrected Sentence: {corrected}")

    # Analyze with spaCy if a correction was made
    doc = nlp(corrected)
    analysis = []

    for token in doc:
        analysis.append({
            "text": token.text,
            "lemma": token.lemma_,
            "pos": token.pos_,
            "tag": token.tag_,
            "dep": token.dep_,
            "explanation": spacy.explain(token.tag_)
        })

    return corrected, analysis

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'wav', 'mp3', 'm4a', 'flac'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def transcribe_audio(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio)
    except Exception as e:
        return f"Error: {str(e)}"

def text_to_audio(text):
    tts = gTTS(text=text, lang='en')
    audio_filename = f"static/response_{random.randint(1000,9999)}.mp3"
    tts.save(audio_filename)
    return audio_filename

@app.route("/", methods=["GET", "POST"])
def index():
    word = None
    vocab_info = None
    phonetic = None
    pronunciation_audio = None
    chat_response = None
    audio_output_path = None
    active_form = "none"
    user_input = None
    score = 0
    feedback = "Excellent"

    vocabulary_fetcher = VocabularyInfoFetcher()

    if request.method == "POST":
        if 'vocabulary' in request.form:
            word = request.form.get("word")
            if word:
                vocab_info = vocabulary_fetcher.get_vocabulary_info(word)
                active_form = "vocabulary"

        elif 'pronunciation' in request.form:
            word = request.form.get("word")
            if word:
                phonetic, pronunciation_audio = get_pronunciation(word)
                active_form = "pronunciation"

        elif 'chat' in request.form:
            user_input = request.form.get("user_input")
            if user_input:
                corrected, analysis = correct_and_analyze(user_input)
                chat_response = f"The correct sentence is '{corrected}'"

                # Calculate score for feedback (mock example)
                score = 0.8  # Placeholder for actual score logic
                if score < 0.25:
                    feedback = "Many mistakes"
                elif score < 0.50:
                    feedback = "Still bad"
                elif score < 0.75:
                    feedback = "Average"
                else:
                    feedback = "Good/Excellent"

                audio_output_path = text_to_audio(chat_response)
                active_form = "chat"

    return render_template("index.html",
                           word=word,
                           vocab_info=vocab_info,
                           phonetic=phonetic,
                           pronunciation_audio=pronunciation_audio,
                           user_input=user_input,
                           chat_response=chat_response,
                           audio_output_path=audio_output_path,
                           feedback=feedback,
                           score=score,
                           active_form="vocabulary" if vocab_info else
                                       "pronunciation" if phonetic else
                                       "chat" if chat_response else "")

if __name__ == "__main__":
    app.run(debug=True)
