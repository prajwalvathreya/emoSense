from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
import torch
import torch.nn.functional as F
import numpy as np
from record_voice import record_audio

model_id = "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3"
model = AutoModelForAudioClassification.from_pretrained(model_id)

feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, do_normalize=True)
id2label = model.config.id2label

def preprocess_audio(audio_array, feature_extractor, max_duration=30.0):
    # audio_array, sampling_rate = librosa.load(audio_path, sr=feature_extractor.sampling_rate)

    max_length = int(feature_extractor.sampling_rate * max_duration)
    if len(audio_array) > max_length:
        audio_array = audio_array[:max_length]
    else:
        audio_array = np.pad(audio_array, (0, max_length - len(audio_array)))

    inputs = feature_extractor(
        audio_array,
        sampling_rate=feature_extractor.sampling_rate,
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    return inputs


def predict_emotion_from_audio(audio_array, model=model, feature_extractor=feature_extractor, id2label=id2label, max_duration=30.0):
    # Preprocess the audio data
    inputs = preprocess_audio(audio_array, feature_extractor, max_duration)

    # Determine the device to use (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Disable gradient calculations for inference
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits

    # Get the probabilities using softmax
    probs = F.softmax(logits, dim=-1)

    # Create a dictionary of emotion probabilities
    emotion_probs = {id2label[i]: probs[0][i].item() for i in range(len(id2label))}

    # Sort emotions based on probability (highest to lowest)
    sorted_emotions = sorted(emotion_probs.items(), key=lambda x: x[1], reverse=True)

    # Get the predicted emotion and its confidence
    predicted_id = torch.argmax(logits, dim=-1).item()
    predicted_emotion = id2label[predicted_id]
    confidence = probs[0][predicted_id].item() * 100  # Convert to percentage

    # Prepare the output dictionary
    result = {
        "emotion": predicted_emotion,
        "confidence": f"{confidence:.2f}%",
        "all_probabilities": {emotion: round(prob, 2) for emotion, prob in sorted_emotions}
    }

    print(f"Predicted Emotion: {result}")

    return result

