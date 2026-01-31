import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from gtts import gTTS
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,   # safer for 8GB
    device_map="auto"
)


def evaluate_soft_skills(answer: str) -> dict:
    prompt = f"""
You are an AI soft-skill coach.

Evaluate the following answer based on:
1. Clarity
2. Confidence
3. Professional tone

Give scores from 0 to 10.

Respond ONLY in valid JSON.
Do not add explanations.

Answer:
{answer}

JSON format:
{{
  "clarity": number,
  "confidence": number,
  "professional_tone": number,
  "feedback": "text",
  "improved_answer": "text"
}}
"""

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=250,
        temperature=0.2,
        do_sample=True
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # ---- SAFE JSON EXTRACTION ----
    start = decoded.find("{")
    end = decoded.rfind("}") + 1

    if start == -1 or end == -1:
        return fallback_response()

    json_str = decoded[start:end]

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return fallback_response()
def fallback_response():
    return {
        "clarity": 5,
        "confidence": 5,
        "professional_tone": 5,
        "feedback": "Your response is understandable but needs clearer structure and stronger confidence.",
        "improved_answer": "I work well in a team by actively supporting others, communicating clearly, and contributing consistently to shared goals."
    }
