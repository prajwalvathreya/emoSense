import os
import json
from openai import OpenAI

token = os.environ["GITHUB_TOKEN"]
endpoint = "https://models.inference.ai.azure.com"
model_name = "gpt-4o"

client = OpenAI(
    base_url=endpoint,
    api_key=token,
)

def compile_emotion(facial_emotion, audio_emotion):
    context_prompt = f"""
    You are an emotion analysis assistant.

    Here is data from two modalities:
    1. Facial Expression Analysis: {facial_emotion}
    2. Audio Voice Analysis: {audio_emotion}

    Your task is to analyze both sources and provide a final emotion judgment by considering:
    - Confidence levels
    - Emotion probability distributions
    - Agreement/disagreement between the modalities

    Return a JSON response like:
    {{
      "final_emotion": "<emotion>",
      "confidence_explanation": "<brief explanation of confidence>"
    }}
    """

    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": context_prompt}
        ],
        temperature=0.7,
        max_tokens=300,
        model=model_name
    )

    try:
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        # Optional fallback: return raw string if JSON parsing fails
        return {"error": "Failed to parse response", "raw_response": response.choices[0].message.content}
