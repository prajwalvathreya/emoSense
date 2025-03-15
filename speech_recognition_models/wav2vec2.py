from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import torch
import librosa
import torch.nn.functional as F

# Load model and feature extractor
model_name = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)

id2label = model.config.id2label

# Load and preprocess audio file
audio_path = "../recordings/english_2025-03-09_15-09-21.wav"
speech, sr = librosa.load(audio_path, sr=16000)  # Convert to 16kHz mono

# Extract features
inputs = feature_extractor(speech, sampling_rate=16000, return_tensors="pt", padding=True)

# Perform inference
with torch.no_grad():
    logits = model(**inputs).logits

# Convert logits to probabilities
probs = F.softmax(logits, dim=-1)

# Get probabilities for each emotion
emotion_probs = {id2label[i]: probs[0][i].item() for i in range(len(id2label))}

# Print sorted results
for emotion, prob in sorted(emotion_probs.items(), key=lambda x: x[1], reverse=True):
    print(f"{emotion}: {prob:.4f}")
