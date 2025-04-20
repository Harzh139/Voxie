import pyttsx3
import os

def get_pronunciation(word):
    phonetic = ' '.join([char.upper() for char in word])
    engine = pyttsx3.init()

    if not os.path.exists("static"):
        os.makedirs("static")

    audio_filename = f"{word}_pronunciation.mp3"
    audio_path = os.path.join("static", audio_filename)

    try:
        engine.save_to_file(word, audio_path)
        engine.runAndWait()
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
            return phonetic, None
    except Exception as e:
        print(f"Error generating pronunciation audio: {e}")
        return phonetic, None

    return phonetic, f"/static/{audio_filename}"
