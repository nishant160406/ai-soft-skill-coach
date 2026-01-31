from ai_engine import evaluate_soft_skills
from gtts import gTTS
from tts_utils import speak_text
text = "I think maybe I am good at teamwork because I sometimes help others."

result = evaluate_soft_skills(text)
print(result)
audio_file = speak_text(result["feedback"])

print("Audio generated:", audio_file)


