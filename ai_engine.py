import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float32,          # fixed deprecation warning
    device_map="auto"
)


def extract_score(text: str, label: str) -> int:
    """
    Extracts the LAST numeric score (0–10) from model output.
    This avoids picking example values.
    """
    pattern = rf"{label}.*?:\s*(\d+)"
    matches = re.findall(pattern, text, re.IGNORECASE)

    if matches:
        score = int(matches[-1])  # take LAST match
        return min(max(score, 0), 10)
    else:
        return 5



def evaluate_soft_skills(answer: str) -> dict:
    prompt =  f"""
You are an AI soft-skill coach.

Here is an example of how to evaluate an answer:

Clarity score (0-10): 6
Confidence score (0-10): 4
Professional tone score (0-10): 6

Feedback:
The response is clear but uses hesitant language, which reduces confidence.

Improved answer:
I am a strong team player who communicates clearly and supports others effectively.

Now evaluate the following answer in the SAME format:

{answer}
"""

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    outputs = model.generate(
    **inputs,
    max_new_tokens=300,
    temperature=0.4,
    do_sample=True,
    repetition_penalty=1.2
)

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("MODEL RAW OUTPUT:\n", decoded)
    return {
        "clarity": extract_score(decoded, "Clarity"),
        "confidence": extract_score(decoded, "Confidence"),
        "professional_tone": extract_score(decoded, "Professional"),
        "feedback": decoded.split("Feedback:")[-1].split("Improved answer:")[0].strip(),
        "improved_answer": decoded.split("Improved answer:")[-1].strip()
    }
    # ---- SAFE JSON EXTRACTION ----
    """ start = decoded.find("{")
    end = decoded.rfind("}") + 1

    if start == -1 or end == -1:
        return fallback_response()

    json_str = decoded[start:end]

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return fallback_response() """