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
    inputs = preprocess_audio(audio_array, feature_extractor, max_duration)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits

    probs = F.softmax(logits, dim=-1)

    emotion_probs = {id2label[i]: probs[0][i].item() for i in range(len(id2label))}

    for emotion, prob in sorted(emotion_probs.items(), key=lambda x: x[1], reverse=True):
        print(f"{emotion}: {prob:.4f}")

    predicted_id = torch.argmax(logits, dim=-1).item()
    predicted_label = id2label[predicted_id]

    return predicted_label

audio_array = record_audio()

predicted_emotion = predict_emotion(audio_array, model, feature_extractor, id2label)
print(f"Predicted Emotion: {predicted_emotion}")
