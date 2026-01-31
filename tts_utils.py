from gtts import gTTS

def speak_text(text, filename="feedback.mp3", lang="en"):
    """
    Converts text to speech and saves as an MP3 file.
    """
    tts = gTTS(text=text, lang=lang)
    tts.save(filename)
    return filename